from __future__ import annotations

import logging
import time
from typing import Any, Dict

from celery.utils.log import get_task_logger
from fastapi import HTTPException

from scperturb_cmap.api.runtime import get_model_path
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.api.settings import get_api_settings
from scperturb_cmap.io.schemas import TargetSignature

from .celery_app import get_celery_app

logger = logging.getLogger(__name__)
task_logger = get_task_logger(__name__)

celery_app = get_celery_app()

__all__ = ["celery_app", "score_target_task"]


@celery_app.task(name="scperturb_cmap.score_target", bind=True)
def score_target_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute connectivity scoring in a background worker.

    The payload is expected to mirror the synchronous scoring request schema.
    """
    settings = get_api_settings()
    start_time = time.time()
    method = str(payload.get("method", "baseline"))

    try:
        target_info = payload["target"]
        target = TargetSignature(
            genes=target_info["genes"],
            weights=target_info["weights"],
            metadata=target_info.get("metadata", {}),
        )
    except KeyError as exc:
        msg = f"Invalid payload for score_target_task: missing field {exc}."
        task_logger.error(msg)
        raise ValueError(msg) from exc

    try:
        model_path = get_model_path(settings, required=method.lower() == "metric")
    except HTTPException as exc:
        # Surface FastAPI HTTP exceptions as task failures; the API layer will expose the message.
        detail = getattr(exc, "detail", str(exc))
        task_logger.error("Pre-flight validation failed: %s", detail)
        raise RuntimeError(detail) from exc

    task_logger.info("Scoring target via Celery (method=%s, top_k=%s).", method, payload.get("top_k"))

    filters = {}
    if payload.get("cell_line"):
        filters["cell_line"] = [payload["cell_line"]]

    result = rank_drugs(
        target_signature=target,
        library=settings.lincs_path,
        method=method,
        model_path=model_path,
        top_k=int(payload.get("top_k", 50)),
        blend=float(payload.get("blend", 0.5)),
        auto_blend=bool(payload.get("auto_blend", False)),
        filters=filters or None,
    )

    execution_time = time.time() - start_time
    metadata = {**result.metadata, "cell_line": payload.get("cell_line")}

    return {
        "method": result.method,
        "ranking": result.ranking.to_dict(orient="records"),
        "metadata": metadata,
        "execution_time": execution_time,
    }
