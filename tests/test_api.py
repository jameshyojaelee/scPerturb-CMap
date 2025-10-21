from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scperturb_cmap.api.runtime import reset_lincs_cache
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
        "CELERY_TASK_ALWAYS_EAGER": "1",
        "CELERY_BROKER_URL": "memory://",
        "CELERY_RESULT_BACKEND": "cache+memory://",
    }

    if create_lincs:
        lincs_path = _write_lincs_fixture(tmp_path / "lincs.csv")
        env["SCPC_LINCS_PATH"] = str(lincs_path)

    if overrides:
        env.update(overrides)

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    reset_api_settings_cache()
    reset_lincs_cache()
    import scripts.api.main as api_main  # noqa: WPS433

    api_module = importlib.reload(api_main)
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


def test_enqueue_job_completes_with_eager_celery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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

    # Patch both synchronous and asynchronous scoring implementations.
    monkeypatch.setattr(api_module, "rank_drugs", lambda **_: stub_result)
    import scperturb_cmap.workers.tasks as worker_tasks  # noqa: WPS433

    monkeypatch.setattr(worker_tasks, "rank_drugs", lambda **_: stub_result)

    payload = {
        "target": {
            "genes": ["CDK1"],
            "weights": [1.0],
            "metadata": {},
        },
        "method": "baseline",
        "top_k": 1,
        "cell_line": "A375",
    }

    with TestClient(api_module.app) as client:
        job_response = client.post("/api/score/jobs", json=payload)

        assert job_response.status_code == 200
        job_data = job_response.json()
        assert job_data["status"] in {"completed", "pending", "running"}

        job_id = job_data["job_id"]
        status_response = client.get(f"/api/score/jobs/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
        else:
            assert status_response.status_code == 202
            status_data = status_response.json()

        assert status_data["job_id"] == job_id
        if status_response.status_code == 200:
            assert status_data["status"] == "completed"
            assert status_data["result"]["method"] == "baseline"

        stream_response = client.get(f"/api/score/jobs/{job_id}/stream")
        assert stream_response.status_code == 200
        lines = [json.loads(line) for line in stream_response.text.strip().splitlines() if line.strip()]
        assert lines[-1]["status"] in {"completed", "failed"}
        if lines[-1]["status"] == "completed":
            assert lines[-1]["result"]["method"] == "baseline"


def test_enqueue_job_returns_503_when_queue_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    overrides = {
        "CELERY_BROKER_URL": "",
        "CELERY_RESULT_BACKEND": "",
        "SCPC_REDIS_URL": "",
    }
    api_module = _load_api_module(monkeypatch, tmp_path, overrides=overrides)

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
        response = client.post("/api/score/jobs", json=payload)

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "http_error"
