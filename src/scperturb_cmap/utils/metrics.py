"""
Metrics collection and monitoring utilities
Integrates with Prometheus, CloudWatch, and Cloud Monitoring
"""
import functools
import os
import time
from typing import Any, Callable, Dict, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """
    Unified metrics collector supporting multiple backends
    """
    
    def __init__(self, backend: str = 'prometheus', port: int = 8000, namespace: str | None = None):
        """
        Initialize metrics collector
        
        Args:
            backend: 'prometheus', 'cloudwatch', 'cloud_monitoring', or 'none'
            port: Port for Prometheus HTTP server
            namespace: Optional namespace for cloud metric backends
        """
        self.backend = backend
        self.enabled = backend != 'none'
        self.namespace = namespace or 'scPerturb-CMap'
        
        if self.backend == 'prometheus' and PROMETHEUS_AVAILABLE:
            self._init_prometheus()
            self._start_prometheus_server(port)
        elif self.backend == 'cloudwatch':
            self._init_cloudwatch()
        elif self.backend == 'cloud_monitoring':
            self._init_cloud_monitoring()
    
    def _init_prometheus(self):
        """Initialize Prometheus metrics"""
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'path', 'status']
        )
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'path']
        )
        self.scoring_duration_seconds = Histogram(
            'scoring_duration_seconds',
            'Scoring operation duration',
            ['method', 'cell_line']
        )
        self.scoring_total = Counter(
            'scoring_total',
            'Total scoring operations',
            ['method', 'status']
        )
        self.active_requests = Gauge(
            'active_requests',
            'Number of active requests'
        )
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        self.data_loaded_bytes = Counter(
            'data_loaded_bytes_total',
            'Total bytes of data loaded',
            ['source']
        )
    
    def _start_prometheus_server(self, port: int):
        """Start Prometheus HTTP server"""
        try:
            start_http_server(port)
            print(f"Prometheus metrics server started on port {port}")
        except Exception as exc:
            print(f"Warning: Failed to start Prometheus server: {exc}")
    
    def _init_cloudwatch(self):
        """Initialize CloudWatch metrics"""
        try:
            import boto3
            self.cloudwatch = boto3.client('cloudwatch')
        except ImportError:
            print("Warning: boto3 not available, CloudWatch metrics disabled")
            self.enabled = False
    
    def _init_cloud_monitoring(self):
        """Initialize Google Cloud Monitoring"""
        try:
            from google.cloud import monitoring_v3
            self.monitoring_client = monitoring_v3.MetricServiceClient()
            project_id = (
                os.environ.get('GOOGLE_CLOUD_PROJECT')
                or os.environ.get('GCP_PROJECT')
                or os.environ.get('GCP_PROJECT_ID')
            )
            if not project_id:
                print(
                    "Warning: GOOGLE_CLOUD_PROJECT or GCP_PROJECT not set; "
                    "Cloud Monitoring metrics disabled"
                )
                self.enabled = False
                return
            self.project_name = self.monitoring_client.common_project_path(project_id)
        except ImportError:
            print("Warning: google-cloud-monitoring not available, metrics disabled")
            self.enabled = False
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"Warning: Failed to initialise Cloud Monitoring: {exc}")
            self.enabled = False
    
    def record_http_request(
        self,
        method: str,
        path: str,
        status: int,
        duration: float
    ):
        """Record HTTP request metrics"""
        if not self.enabled:
            return
        
        if self.backend == 'prometheus' and PROMETHEUS_AVAILABLE:
            self.http_requests_total.labels(
                method=method,
                path=path,
                status=str(status)
            ).inc()
            self.http_request_duration_seconds.labels(
                method=method,
                path=path
            ).observe(duration)
        elif self.backend == 'cloudwatch':
            self._put_cloudwatch_metrics([
                {
                    'MetricName': 'RequestCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Method', 'Value': method},
                        {'Name': 'Path', 'Value': path},
                        {'Name': 'Status', 'Value': str(status)}
                    ]
                },
                {
                    'MetricName': 'RequestDuration',
                    'Value': duration,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Method', 'Value': method},
                        {'Name': 'Path', 'Value': path}
                    ]
                }
            ])
    
    def record_scoring_operation(
        self,
        method: str,
        cell_line: Optional[str],
        duration: float,
        success: bool
    ):
        """Record scoring operation metrics"""
        if not self.enabled:
            return
        
        if self.backend == 'prometheus' and PROMETHEUS_AVAILABLE:
            self.scoring_duration_seconds.labels(
                method=method,
                cell_line=cell_line or 'all'
            ).observe(duration)
            self.scoring_total.labels(
                method=method,
                status='success' if success else 'failure'
            ).inc()
        elif self.backend == 'cloudwatch':
            self._put_cloudwatch_metrics([
                {
                    'MetricName': 'ScoringDuration',
                    'Value': duration,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Method', 'Value': method},
                        {'Name': 'CellLine', 'Value': cell_line or 'all'}
                    ]
                },
                {
                    'MetricName': 'ScoringCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Method', 'Value': method},
                        {'Name': 'Status', 'Value': 'success' if success else 'failure'}
                    ]
                }
            ])
    
    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache access metrics"""
        if not self.enabled:
            return
        
        if self.backend == 'prometheus' and PROMETHEUS_AVAILABLE:
            if hit:
                self.cache_hits.labels(cache_type=cache_type).inc()
            else:
                self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_data_load(self, source: str, bytes_loaded: int):
        """Record data loading metrics"""
        if not self.enabled:
            return
        
        if self.backend == 'prometheus' and PROMETHEUS_AVAILABLE:
            self.data_loaded_bytes.labels(source=source).inc(bytes_loaded)
    
    def _put_cloudwatch_metrics(self, metrics: list):
        """Put metrics to CloudWatch"""
        if not hasattr(self, 'cloudwatch'):
            return
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics
            )
        except Exception as exc:
            print(f"Warning: Failed to put CloudWatch metrics: {exc}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(
    backend: Optional[str] = None,
    port: Optional[int] = None,
    namespace: Optional[str] = None,
) -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        import os
        resolved_backend = backend or os.environ.get('SCPC_METRICS_BACKEND') or os.environ.get('METRICS_BACKEND', 'prometheus')
        resolved_port = int(
            port
            or os.environ.get('SCPC_METRICS_PORT')
            or os.environ.get('METRICS_PORT', '8000')
        )
        resolved_namespace = namespace or os.environ.get('SCPC_METRICS_NAMESPACE')
        _metrics_collector = MetricsCollector(
            backend=resolved_backend,
            port=resolved_port,
            namespace=resolved_namespace,
        )
    return _metrics_collector


def track_time(metric_name: str = None, labels: Dict[str, str] = None):
    """
    Decorator to track function execution time
    
    Usage:
        @track_time('scoring_duration')
        def score_compounds(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metrics
                if metric_name == 'scoring_duration':
                    method = kwargs.get('method', 'baseline')
                    cell_line = kwargs.get('cell_line')
                    collector.record_scoring_operation(
                        method=method,
                        cell_line=cell_line,
                        duration=duration,
                        success=success
                    )
        
        return wrapper
    return decorator
