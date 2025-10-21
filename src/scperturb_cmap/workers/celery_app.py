from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from celery import Celery

from scperturb_cmap.api.settings import get_api_settings


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_celery_app() -> Celery:
    """Create (or return cached) Celery application configured from environment."""
    settings = get_api_settings()

    broker_url = os.getenv("CELERY_BROKER_URL") or settings.redis_url or "redis://localhost:6379/0"
    result_backend = (
        os.getenv("CELERY_RESULT_BACKEND")
        or os.getenv("CELERY_BROKER_URL")
        or settings.redis_url
        or "redis://localhost:6379/1"
    )

    app = Celery("scperturb_cmap", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_default_queue="scperturb_cmap",
    )

    if _parse_bool(os.getenv("CELERY_TASK_ALWAYS_EAGER")):
        # Useful for local development and unit tests.
        app.conf.task_always_eager = True
        app.conf.task_store_eager_result = True

    return app
