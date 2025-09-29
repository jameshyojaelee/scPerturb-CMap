"""
Cloud-optimized storage utilities for LINCS data
Implements efficient Parquet partitioning strategies for AWS S3 and GCP GCS
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs


class CloudParquetPartitioner:
    """
    Manages cloud-optimized Parquet partitioning strategies for LINCS data
    
    Partitioning strategies:
    1. By cell_line: Optimizes cell-line specific queries
    2. By compound: Optimizes compound-specific lookups
    3. By date/batch: Optimizes temporal queries
    4. Hybrid: Combines multiple partition keys
    """
    
    PARTITION_STRATEGIES = {
        'cell_line': ['cell_line'],
        'compound': ['compound'],
        'cell_line_compound': ['cell_line', 'compound'],
        'batch': ['batch_id'],
        'date': ['year', 'month'],
    }
    
    def __init__(
        self,
        cloud_provider: str = 'auto',
        bucket: Optional[str] = None,
        region: Optional[str] = None
    ):
        """
        Initialize cloud storage manager
        
        Args:
            cloud_provider: 'aws', 'gcp', or 'auto' (detect from environment)
            bucket: S3 bucket or GCS bucket name
            region: AWS region or GCP region
        """
        self.cloud_provider = self._detect_provider(cloud_provider)
        self.bucket = bucket or self._get_default_bucket()
        self.region = region or self._get_default_region()
        self.filesystem = self._init_filesystem()
    
    def _detect_provider(self, provider: str) -> str:
        """Detect cloud provider from environment"""
        if provider != 'auto':
            return provider
        
        # Check for AWS environment
        if 'AWS_REGION' in os.environ or 'AWS_DEFAULT_REGION' in os.environ:
            return 'aws'
        
        # Check for GCP environment
        if 'GCP_PROJECT' in os.environ or 'GOOGLE_CLOUD_PROJECT' in os.environ:
            return 'gcp'
        
        # Default to local filesystem
        return 'local'
    
    def _get_default_bucket(self) -> Optional[str]:
        """Get default bucket from environment"""
        if self.cloud_provider == 'aws':
            return os.environ.get('LINCS_BUCKET', 'scperturb-cmap-lincs-data')
        elif self.cloud_provider == 'gcp':
            return os.environ.get('LINCS_BUCKET', 'scperturb-cmap-lincs-data')
        return None
    
    def _get_default_region(self) -> Optional[str]:
        """Get default region from environment"""
        if self.cloud_provider == 'aws':
            return os.environ.get('AWS_REGION', 'us-east-1')
        elif self.cloud_provider == 'gcp':
            return os.environ.get('GCP_REGION', 'us-central1')
        return None
    
    def _init_filesystem(self) -> fs.FileSystem:
        """Initialize PyArrow filesystem for cloud storage"""
        if self.cloud_provider == 'aws':
            return fs.S3FileSystem(
                region=self.region,
                endpoint_override=os.environ.get('S3_ENDPOINT')
            )
        elif self.cloud_provider == 'gcp':
            return fs.GcsFileSystem()
        else:
            return fs.LocalFileSystem()
    
    def partition_dataset(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        strategy: str = 'cell_line',
        max_rows_per_file: int = 100000,
        compression: str = 'snappy',
        **kwargs
    ) -> Dict[str, int]:
        """
        Partition a LINCS dataset using the specified strategy
        
        Args:
            input_path: Path to input Parquet file or directory
            output_path: Output path for partitioned dataset
            strategy: Partitioning strategy (see PARTITION_STRATEGIES)
            max_rows_per_file: Maximum rows per partition file
            compression: Compression codec ('snappy', 'gzip', 'zstd')
            **kwargs: Additional partitioning parameters
        
        Returns:
            Dictionary with partition statistics
        """
        if strategy not in self.PARTITION_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: {list(self.PARTITION_STRATEGIES.keys())}"
            )
        
        partition_cols = self.PARTITION_STRATEGIES[strategy]
        
        # Read input dataset
        print(f"Reading dataset from {input_path}")
        if self.cloud_provider in ('aws', 'gcp'):
            input_uri = f"{self.bucket}/{input_path}"
            dataset = ds.dataset(input_uri, filesystem=self.filesystem)
        else:
            dataset = ds.dataset(input_path)
        
        # Convert to table for partitioning
        table = dataset.to_table()
        print(f"Loaded {table.num_rows:,} rows, {table.num_columns} columns")
        
        # Add partition columns if needed (e.g., year/month from date)
        if strategy == 'date':
            table = self._add_date_partitions(table, **kwargs)
        
        # Validate partition columns exist
        missing_cols = [col for col in partition_cols if col not in table.column_names]
        if missing_cols:
            raise ValueError(f"Missing partition columns: {missing_cols}")
        
        # Write partitioned dataset
        print(f"Writing partitioned dataset to {output_path}")
        if self.cloud_provider in ('aws', 'gcp'):
            output_uri = f"{self.bucket}/{output_path}"
        else:
            output_uri = str(output_path)
        
        pq.write_to_dataset(
            table,
            root_path=output_uri,
            partition_cols=partition_cols,
            filesystem=self.filesystem,
            max_rows_per_file=max_rows_per_file,
            compression=compression,
            existing_data_behavior='overwrite_or_ignore',
            use_dictionary=True,
            write_statistics=True,
            version='2.6'  # Use Parquet format version 2.6 for better compression
        )
        
        # Collect statistics
        partitions = self._count_partitions(output_uri)
        stats = {
            'total_rows': table.num_rows,
            'num_partitions': len(partitions),
            'partition_strategy': strategy,
            'compression': compression,
            'partitions': partitions
        }
        
        print(f"Partitioning complete: {len(partitions)} partitions created")
        return stats
    
    def _add_date_partitions(self, table: pa.Table, date_col: str = 'date') -> pa.Table:
        """Add year/month partition columns from date column"""
        if date_col not in table.column_names:
            raise ValueError(f"Date column '{date_col}' not found in table")
        
        # Convert to pandas for date extraction
        df = table.to_pandas()
        df['year'] = pd.to_datetime(df[date_col]).dt.year
        df['month'] = pd.to_datetime(df[date_col]).dt.month
        
        return pa.Table.from_pandas(df)
    
    def _count_partitions(self, path: str) -> Dict[str, int]:
        """Count rows per partition"""
        try:
            dataset = ds.dataset(path, filesystem=self.filesystem)
            partitions = {}
            for fragment in dataset.get_fragments():
                partition_expr = str(fragment.partition_expression)
                num_rows = fragment.metadata.num_rows
                partitions[partition_expr] = num_rows
            return partitions
        except Exception as e:
            print(f"Warning: Could not count partitions: {e}")
            return {}
    
    def read_partitioned(
        self,
        path: Union[str, Path],
        filters: Optional[List] = None,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read partitioned dataset with predicate pushdown
        
        Args:
            path: Path to partitioned dataset
            filters: PyArrow filters for partition pruning
                Example: [('cell_line', '=', 'A549')]
            columns: Columns to read (projection pushdown)
            **kwargs: Additional read parameters
        
        Returns:
            DataFrame with filtered data
        """
        if self.cloud_provider in ('aws', 'gcp'):
            uri = f"{self.bucket}/{path}"
        else:
            uri = str(path)
        
        print(f"Reading partitioned dataset from {uri}")
        dataset = ds.dataset(uri, filesystem=self.filesystem)
        
        # Apply filters and column selection
        table = dataset.to_table(
            filter=ds.field(*filters[0]) if filters and len(filters) == 1 else None,
            columns=columns
        )
        
        print(f"Read {table.num_rows:,} rows after filtering")
        return table.to_pandas()


class CloudDataCache:
    """
    Local caching layer for cloud-stored LINCS data
    Implements LRU cache with configurable size limits
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = '/tmp/scperturb_cache',
        max_cache_size_gb: float = 10.0
    ):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Local directory for cache
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024**3  # Convert to bytes
    
    def get(self, key: str, remote_path: str, filesystem: fs.FileSystem) -> Path:
        """
        Get file from cache or download from cloud
        
        Args:
            key: Cache key (usually filename)
            remote_path: Remote path in cloud storage
            filesystem: PyArrow filesystem
        
        Returns:
            Path to cached file
        """
        cache_path = self.cache_dir / key
        
        # Check if cached and valid
        if cache_path.exists():
            print(f"Cache hit: {key}")
            return cache_path
        
        # Download from cloud
        print(f"Cache miss: downloading {key} from {remote_path}")
        with filesystem.open_input_file(remote_path) as remote_file:
            data = remote_file.read()
            cache_path.write_bytes(data)
        
        # Evict old entries if cache is full
        self._evict_if_needed()
        
        return cache_path
    
    def _evict_if_needed(self):
        """Evict oldest files if cache exceeds size limit"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.iterdir())
        
        if total_size > self.max_cache_size:
            print(f"Cache size {total_size/1024**3:.2f}GB exceeds limit, evicting...")
            
            # Sort files by access time
            files = sorted(
                self.cache_dir.iterdir(),
                key=lambda f: f.stat().st_atime
            )
            
            # Evict oldest files until under limit
            for file in files:
                if total_size <= self.max_cache_size * 0.8:  # 80% threshold
                    break
                file_size = file.stat().st_size
                file.unlink()
                total_size -= file_size
                print(f"Evicted: {file.name} ({file_size/1024**2:.2f}MB)")
    
    def clear(self):
        """Clear all cached files"""
        for file in self.cache_dir.iterdir():
            file.unlink()
        print(f"Cleared cache: {self.cache_dir}")


# Convenience functions
def partition_lincs_for_cloud(
    input_path: str,
    output_path: str,
    strategy: str = 'cell_line',
    cloud_provider: str = 'auto',
    **kwargs
) -> Dict[str, int]:
    """
    Partition LINCS dataset for cloud storage
    
    Args:
        input_path: Input Parquet path
        output_path: Output path for partitioned data
        strategy: Partitioning strategy
        cloud_provider: 'aws', 'gcp', or 'auto'
        **kwargs: Additional arguments for partitioner
    
    Returns:
        Partition statistics
    """
    partitioner = CloudParquetPartitioner(cloud_provider=cloud_provider)
    return partitioner.partition_dataset(input_path, output_path, strategy, **kwargs)


def load_lincs_from_cloud(
    path: str,
    cell_line: Optional[str] = None,
    cloud_provider: str = 'auto',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load LINCS data from cloud storage with filtering
    
    Args:
        path: Path to partitioned dataset
        cell_line: Filter by cell line
        cloud_provider: 'aws', 'gcp', or 'auto'
        columns: Columns to load
    
    Returns:
        Filtered DataFrame
    """
    partitioner = CloudParquetPartitioner(cloud_provider=cloud_provider)
    filters = [('cell_line', '=', cell_line)] if cell_line else None
    return partitioner.read_partitioned(path, filters=filters, columns=columns)
