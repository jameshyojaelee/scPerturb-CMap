from __future__ import annotations

from pydantic import BaseModel, Field

try:
    # Lazy import in type: ignore context to avoid hard dependency at import time
    from .utils.device import get_device
except Exception:  # pragma: no cover - if anything goes wrong, default at runtime
    def get_device() -> str:  # type: ignore
        return "cpu"


class AppConfig(BaseModel):
    """Basic application configuration.

    This is intentionally minimal and safe at import time. The device is
    determined at instantiation using the helper.
    """

    seed: int = 42
    device: str = Field(default_factory=get_device)

