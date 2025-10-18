"""
Celery worker bootstrap for scPerturb-CMap.

Provides a minimal task surface so infrastructure files can start workers
without failing when Redis/queues are healthy but no workloads have been
scheduled yet. The implementation deliberately keeps the default broker/backend
configurable via environment variables so deployments can override them.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from celery import Celery

logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL)
DEFAULT_QUEUE = os.environ.get("CELERY_DEFAULT_QUEUE", "scperturb-cmap")

celery_app = Celery(
    "scperturb_cmap",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)
celery_app.conf.task_default_queue = DEFAULT_QUEUE

# Expose the Celery instance as `app` so `celery -A scperturb_cmap.worker worker`
# picks it up without additional flags.
app = celery_app


@celery_app.task(name="scperturb_cmap.tasks.ping")
def ping() -> str:
    """Simple health-check task used by smoke tests."""
    return "pong"


@celery_app.task(name="scperturb_cmap.tasks.score_target")
def score_target(
    payload: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Placeholder scoring task.

    For now this simply reports that work has been accepted so the worker
    infrastructure can be validated independently of the heavy scoring logic.
    Future revisions can plug into rank_drugs or asynchronous pipelines.
    """
    genes = payload.get("genes")
    gene_count = len(genes) if isinstance(genes, list) else 0
    logger.info("Received score task for %s genes (options=%s)", gene_count, options)
    return {
        "status": "accepted",
        "gene_count": gene_count,
        "options": options or {},
    }


__all__ = ["app", "celery_app", "ping", "score_target"]
