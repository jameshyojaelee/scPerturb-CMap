# Roadmap

This roadmap sketches the near-term direction for scPerturb-CMap. As priorities
shift with collaborator feedback, expect updates roughly once per quarter.

## Near-Term (0–3 months)

- Harden the public API: configurable readiness checks, rate limiting, and
  improved error reporting for missing datasets or models.
- Expand documentation with end-to-end notebooks covering target construction,
  scoring, and explainability for common disease archetypes.
- Finalise the reproducible demo pipeline so `make acceptance` runs cleanly on
  fresh machines and publishes summary metrics.
- Improve explainability output ergonomics—compact plots, richer narratives,
  and streamlined CSV/Parquet exports from the Streamlit UI.

## Mid-Term (3–6 months)

- Ship a lightweight web deployment profile (Docker Compose + SQLite/Redis) for
  small teams, alongside production Helm charts.
- Introduce asynchronous scoring queues backed by Celery, including retry
  policies and progress tracking endpoints.
- Validate support for user-supplied LINCS-like libraries, covering unit tests,
  schema validation, and benchmarking utilities.
- Integrate automated benchmarking against public perturbation datasets to
  track regression performance across releases.

## Stretch Goals

- Add optional model fine-tuning workflows that ingest user-provided inversion
  pairs while safeguarding against catastrophic forgetting.
- Explore GPU-accelerated inference paths for the DualEncoder to support large
  batch scoring workloads.
- Publish fully hosted docs (Read the Docs or GitHub Pages) with versioned API
  references generated from docstrings.
- Provide turnkey telemetry dashboards (Grafana/Prometheus) and alerts tailored
  for common deployment targets (AWS/GCP/Kubernetes).

If you have feature requests or would like to collaborate, open an issue or
reach out via the project Slack.
