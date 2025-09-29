"""
Google Cloud Functions handler for scPerturb-CMap serverless scoring API
Optimized for Cloud Storage and BigQuery integration
"""
import json
import os
import time
from typing import Any, Dict

import functions_framework
import pandas as pd
from google.cloud import storage
from flask import Request, jsonify

# Lazy imports for cold start optimization
_rank_drugs = None
_TargetSignature = None


def lazy_import_scoring():
    """Lazy import of scoring modules to reduce cold start time"""
    global _rank_drugs, _TargetSignature
    if _rank_drugs is None:
        from scperturb_cmap.api.score import rank_drugs
        from scperturb_cmap.io.schemas import TargetSignature
        _rank_drugs = rank_drugs
        _TargetSignature = TargetSignature


# Configuration from environment
PROJECT_ID = os.environ.get('GCP_PROJECT')
LINCS_BUCKET = os.environ.get('LINCS_BUCKET', 'scperturb-cmap-lincs-data')
LINCS_BLOB = os.environ.get('LINCS_BLOB', 'lincs_level5_landmark_long.parquet')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'scperturb-cmap-models')
MODEL_BLOB = os.environ.get('MODEL_BLOB', 'best.pt')

# Initialize GCS client
storage_client = storage.Client()


def download_from_gcs(bucket_name: str, blob_name: str, local_path: str) -> bool:
    """Download file from GCS with error handling"""
    try:
        if os.path.exists(local_path):
            # Check if file is recent (within 1 hour)
            if time.time() - os.path.getmtime(local_path) < 3600:
                print(f"Using cached file: {local_path}")
                return True
        
        print(f"Downloading gs://{bucket_name}/{blob_name} to {local_path}")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"Error downloading from GCS: {e}")
        return False


def load_lincs_library(cell_line: str = None) -> pd.DataFrame:
    """Load LINCS library from GCS with optional cell line filtering"""
    local_path = '/tmp/lincs_library.parquet'
    
    # Download if not cached
    if not download_from_gcs(LINCS_BUCKET, LINCS_BLOB, local_path):
        raise RuntimeError("Failed to download LINCS library from GCS")
    
    # Load with PyArrow for efficient partitioned reading
    df = pd.read_parquet(local_path, engine='pyarrow')
    
    # Filter by cell line if specified
    if cell_line and 'cell_line' in df.columns:
        df = df[df['cell_line'] == cell_line]
        print(f"Filtered to {len(df)} rows for cell line: {cell_line}")
    
    return df


@functions_framework.http
def score(request: Request):
    """
    Cloud Function for scoring requests
    
    Expected request body:
    {
        "target": {
            "genes": ["GENE1", "GENE2", ...],
            "weights": [1.5, -2.3, ...],
            "metadata": {...}
        },
        "method": "baseline" or "metric",
        "top_k": 50,
        "cell_line": "A549" (optional),
        "blend": 0.5 (optional, for metric method)
    }
    """
    start_time = time.time()
    
    # Set CORS headers
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    try:
        # Lazy import scoring modules
        lazy_import_scoring()
        
        # Parse request
        request_json = request.get_json(silent=True)
        if not request_json:
            return (jsonify({'error': 'Invalid JSON'}), 400, headers)
        
        # Extract parameters
        target_data = request_json.get('target')
        method = request_json.get('method', 'baseline')
        top_k = request_json.get('top_k', 50)
        cell_line = request_json.get('cell_line')
        blend = request_json.get('blend', 0.5)
        
        if not target_data:
            return (jsonify({'error': 'Missing target signature'}), 400, headers)
        
        # Create target signature
        target = _TargetSignature(
            genes=target_data['genes'],
            weights=target_data['weights'],
            metadata=target_data.get('metadata', {})
        )
        
        # Load LINCS library
        print(f"Loading LINCS library (cell_line={cell_line})")
        library = load_lincs_library(cell_line=cell_line)
        
        # Download model if using metric method
        model_path = None
        if method == 'metric':
            model_path = '/tmp/model.pt'
            if not download_from_gcs(MODEL_BUCKET, MODEL_BLOB, model_path):
                return (jsonify({'error': 'Failed to load model from GCS'}), 500, headers)
        
        # Perform scoring
        print(f"Scoring with method={method}, top_k={top_k}")
        result = _rank_drugs(
            target_signature=target,
            library=library,
            method=method,
            model_path=model_path,
            top_k=top_k,
            blend=blend
        )
        
        # Convert result to JSON-serializable format
        ranking_df = result.ranking
        ranking_json = ranking_df.to_dict(orient='records')
        
        elapsed_time = time.time() - start_time
        
        response_body = {
            'method': result.method,
            'ranking': ranking_json,
            'metadata': {
                **result.metadata,
                'execution_time_seconds': elapsed_time,
                'num_results': len(ranking_json)
            }
        }
        
        return (jsonify(response_body), 200, headers)
        
    except Exception as e:
        print(f"Error in cloud function: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return (jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500, headers)


@functions_framework.http
def health(request: Request):
    """Health check endpoint"""
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    if request.method == 'OPTIONS':
        headers['Access-Control-Allow-Methods'] = 'GET'
        headers['Access-Control-Allow-Headers'] = 'Content-Type'
        headers['Access-Control-Max-Age'] = '3600'
        return ('', 204, headers)
    
    return (jsonify({
        'status': 'healthy',
        'service': 'scperturb-cmap-scoring',
        'version': '0.2.0'
    }), 200, headers)
