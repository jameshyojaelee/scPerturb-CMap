# API Reference

This guide summarises the public Python interface exposed by scPerturb-CMap.
Each section includes a short snippet you can adapt in notebooks or scripts.

## Scoring (`scperturb_cmap.api.score`)

```python
from pathlib import Path

import pandas as pd

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.io.schemas import TargetSignature

library = load_lincs_long("examples/data/lincs_demo.parquet")
target = TargetSignature.model_validate_json(Path("examples/out/target.json").read_text())

result = rank_drugs(target_signature=target, library=library, method="baseline", top_k=25)
result.ranking.head()
```

Key arguments:

- `target_signature`: validated `TargetSignature`
- `library`: long-form LINCS DataFrame (or `(matrix, genes, meta)` tuple)
- `method`: `"baseline"` or `"metric"` (supply `model_path` for metric)
- `top_k`: number of rows to return
- `auto_blend` / `blend`: blending weights when mixing baseline + metric

The return value is a `ScoreResult` containing a ranking DataFrame plus metadata
such as the blend factor.

## Explainability (`scperturb_cmap.api.explain`) {#explainability}

```python
from scperturb_cmap.api.explain import ExplainabilityEngine

engine = ExplainabilityEngine(enable_pathway_enrichment=False)
explained = engine.explain_top_k_drugs(
    target_signature=target,
    score_result=result,
    library=library,
    top_k=10,
)

explained[["compound", "score", "narrative"]].head()
```

Core helpers:

- `ExplainabilityEngine.explain_ranking` – deep dive for a single compound
- `ExplainabilityEngine.explain_top_k_drugs` – batch explanations with narratives
- `ExplainabilityEngine.compare_drugs` – side-by-side comparison of two hits

The explainability modules rely on the raw long-form library to recover gene
scores, so reuse the same DataFrame passed to `rank_drugs`.

## Schemas (`scperturb_cmap.io.schemas`)

Use the Pydantic models to validate JSON payloads exchanged between the CLI,
API, and UI.

```python
from scperturb_cmap.io.schemas import TargetSignature, ScoreResult

payload = TargetSignature(
    genes=["G1", "G2", "G10"],
    weights=[1.0, 1.0, -1.0],
    metadata={"case": "demo"},
)

result = ScoreResult(
    method="baseline",
    ranking=result.ranking,  # any DataFrame with required columns
    metadata=result.metadata,
)
```

`TargetSignature` ensures genes/weights align and remain finite, while
`ScoreResult` validates that the ranking exposes the expected columns
(`signature_id`, `compound`, `cell_line`, `score`, `moa`, `target`).

## CLI Helpers (`scperturb_cmap.cli`)

The CLI is built with Typer. You can invoke commands programmatically using the
same module:

```python
from typer.testing import CliRunner

from scperturb_cmap.cli import app

runner = CliRunner()
result = runner.invoke(
    app,
    [
        "score",
        "--target-json",
        "examples/out/target.json",
        "--library",
        "examples/data/lincs_demo.parquet",
        "--method",
        "baseline",
        "--top-k",
        "10",
    ],
)
print(result.exit_code)
```

For the full set of commands and options, see the [CLI reference](cli.md).
