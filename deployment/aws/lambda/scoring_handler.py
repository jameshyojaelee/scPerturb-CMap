"""
AWS Lambda handler for scPerturb-CMap serverless scoring API
Optimized for cold starts with lazy loading and S3 caching
"""
import json
import os
import time
from typing import Any, Dict

import boto3
import pandas as pd
from botocore.exceptions import ClientError

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


# Initialize AWS clients
s3_client = boto3.client('s3')

# Configuration from environment
LINCS_BUCKET = os.environ.get('LINCS_BUCKET', 'scperturb-cmap-lincs-data')
LINCS_KEY = os.environ.get('LINCS_KEY', 'lincs_level5_landmark_long.parquet')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'scperturb-cmap-models')
MODEL_KEY = os.environ.get('MODEL_KEY', 'best.pt')
CACHE_DIR = '/tmp/scperturb_cache'

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)


def download_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """Download file from S3 with error handling"""
    try:
        if os.path.exists(local_path):
            # Check if file is recent (within 1 hour for Lambda tmp persistence)
            if time.time() - os.path.getmtime(local_path) < 3600:
                print(f"Using cached file: {local_path}")
                return True
        
        print(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        return True
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return False


def load_lincs_library(cell_line: str = None) -> pd.DataFrame:
    """Load LINCS library from S3 with optional cell line filtering"""
    local_path = os.path.join(CACHE_DIR, 'lincs_library.parquet')
    
    # Download if not cached
    if not download_from_s3(LINCS_BUCKET, LINCS_KEY, local_path):
        raise RuntimeError("Failed to download LINCS library from S3")
    
    # Load with PyArrow for efficient partitioned reading
    df = pd.read_parquet(local_path, engine='pyarrow')
    
    # Filter by cell line if specified
    if cell_line and 'cell_line' in df.columns:
        df = df[df['cell_line'] == cell_line]
        print(f"Filtered to {len(df)} rows for cell line: {cell_line}")
    
    return df


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for scoring requests
    
    Expected event structure:
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
    
    try:
        # Lazy import scoring modules
        lazy_import_scoring()
        
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        # Extract parameters
        target_data = body.get('target')
        method = body.get('method', 'baseline')
        top_k = body.get('top_k', 50)
        cell_line = body.get('cell_line')
        blend = body.get('blend', 0.5)
        
        if not target_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing target signature'})
            }
        
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
            model_path = os.path.join(CACHE_DIR, 'model.pt')
            if not download_from_s3(MODEL_BUCKET, MODEL_KEY, model_path):
                return {
                    'statusCode': 500,
                    'body': json.dumps({'error': 'Failed to load model from S3'})
                }
        
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
                'num_results': len(ranking_json),
                'lambda_request_id': context.request_id if context else None
            }
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'X-Execution-Time': str(elapsed_time)
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        print(f"Error in lambda handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'type': type(e).__name__
            })
        }


def health_check_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'healthy',
            'service': 'scperturb-cmap-scoring',
            'version': '0.2.0'
        })
    }
