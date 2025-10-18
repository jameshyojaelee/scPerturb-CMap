from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scperturb_cmap.api.settings import get_api_settings, reset_api_settings_cache
from scperturb_cmap.io.schemas import ScoreResult


def _write_lincs_fixture(path: Path) -> Path:
    df = pd.DataFrame(
        {
            "signature_id": ["sig1", "sig1", "sig2", "sig2"],
            "compound": ["cmpd1", "cmpd1", "cmpd2", "cmpd2"],
            "cell_line": ["A375", "A375", "MCF7", "MCF7"],
            "gene_symbol": ["CDK1", "EGFR", "CDK1", "EGFR"],
            "score": [1.0, -1.0, 0.5, -0.5],
            "moa": ["kinase", "kinase", "kinase", "kinase"],
            "target": ["MAPK", "MAPK", "MAPK", "MAPK"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _load_api_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: Optional[Dict[str, str]] = None,
    *,
    create_lincs: bool = True,
):
    env: Dict[str, str] = {
        "SCPC_ENV": "test",
        "SCPC_METRICS_BACKEND": "none",
        "SCPC_CORS_ORIGINS": "http://testserver",
        "SCPC_CACHE_TTL": "0",
        "SCPC_REQUEST_TIMEOUT": "30",
    }

    if create_lincs:
        lincs_path = _write_lincs_fixture(tmp_path / "lincs.csv")
        env["SCPC_LINCS_PATH"] = str(lincs_path)

    if overrides:
        env.update(overrides)

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    reset_api_settings_cache()
    import scripts.api.main as api_main  # noqa: WPS433

    api_module = importlib.reload(api_main)
    api_module._load_lincs_cached.cache_clear()  # type: ignore[attr-defined]
    return api_module


def test_api_settings_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCPC_ENV", "development")
    monkeypatch.setenv("SCPC_CORS_ORIGINS", '["https://example.org","https://foo.bar"]')
    monkeypatch.setenv("SCPC_MAX_REQUEST_BYTES", "1024")
    monkeypatch.setenv("SCPC_METRICS_BACKEND", "cloudwatch")
    reset_api_settings_cache()

    settings = get_api_settings()

    assert settings.is_development is True
    assert settings.cors_origins == ["https://example.org", "https://foo.bar"]
    assert settings.max_request_bytes == 1024
    assert settings.metrics_backend == "cloudwatch"


def test_readiness_reports_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    api_module = _load_api_module(monkeypatch, tmp_path)
    with TestClient(api_module.app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["lincs"]["status"] == "ok"
    assert data["checks"]["model"]["status"] == "skipped"


def test_readiness_reports_missing_lincs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    missing = tmp_path / "missing.csv"
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        overrides={"SCPC_LINCS_PATH": str(missing)},
    )

    with TestClient(api_module.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unready"
    assert any(err.startswith("lincs") for err in data["errors"])


def test_score_rejects_large_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    api_module = _load_api_module(
        monkeypatch,
        tmp_path,
        overrides={"SCPC_MAX_REQUEST_BYTES": "200"},
    )
    big_note = "x" * 1024
    payload = {
        "target": {
            "genes": ["CDK1"],
            "weights": [1.0],
            "metadata": {"note": big_note},
        },
        "method": "baseline",
        "top_k": 5,
    }

    with TestClient(api_module.app) as client:
        response = client.post("/api/score", json=payload)

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "http_error"


def test_score_success_with_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    api_module = _load_api_module(monkeypatch, tmp_path)

    ranking_df = pd.DataFrame(
        [
            {
                "signature_id": "sig1",
                "compound": "cmpd1",
                "cell_line": "A375",
                "score": -1.23,
                "moa": "kinase",
                "target": "MAPK",
            }
        ]
    )
    stub_result = ScoreResult(method="baseline", ranking=ranking_df, metadata={"foo": "bar"})
    monkeypatch.setattr(api_module, "rank_drugs", lambda **_: stub_result)

    payload = {
        "target": {
            "genes": ["CDK1"],
            "weights": [1.0],
            "metadata": {},
        },
        "method": "baseline",
        "top_k": 1,
    }

    with TestClient(api_module.app) as client:
        response = client.post("/api/score", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "baseline"
    assert data["metadata"]["foo"] == "bar"
