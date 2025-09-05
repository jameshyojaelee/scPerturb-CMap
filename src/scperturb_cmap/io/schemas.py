from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

REQUIRED_RANKING_COLUMNS = [
    "signature_id",
    "compound",
    "cell_line",
    "score",
    "moa",
    "target",
]


def _as_list_of_floats(values: Any) -> List[float]:
    if isinstance(values, np.ndarray):
        return [float(x) for x in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [float(x) for x in values]
    raise TypeError("Expected a list/tuple or numpy.ndarray of floats")


class TargetSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genes: List[str] = Field(...)
    weights: List[float] = Field(..., description="Weights aligned to genes order")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("genes")
    @classmethod
    def _validate_genes(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list) or not v:
            raise ValueError("genes must be a non-empty list of strings")
        if not all(isinstance(g, str) for g in v):
            raise ValueError("genes must contain only strings")
        return v

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, v: Any) -> List[float]:
        return _as_list_of_floats(v)

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("weights must be non-empty and align with genes")
        if not all(np.isfinite(vv) for vv in v):
            raise ValueError("weights must be finite numbers")
        return v

    @model_validator(mode="after")
    def _check_alignment(self) -> "TargetSignature":
        if len(self.genes) != len(self.weights):
            raise ValueError("genes and weights must be the same length")
        return self


class DrugSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signature_id: str
    compound: str
    cell_line: Optional[str] = None
    genes: List[str]
    values: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("genes")
    @classmethod
    def _validate_genes(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list) or not v:
            raise ValueError("genes must be a non-empty list of strings")
        if not all(isinstance(g, str) for g in v):
            raise ValueError("genes must contain only strings")
        return v

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v: Any) -> List[float]:
        return _as_list_of_floats(v)

    @field_validator("values")
    @classmethod
    def _validate_values(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("values must be non-empty and align with genes")
        if not all(np.isfinite(vv) for vv in v):
            raise ValueError("values must be finite numbers")
        return v

    @model_validator(mode="after")
    def _check_alignment(self) -> "DrugSignature":
        if len(self.genes) != len(self.values):
            raise ValueError("genes and values must be the same length")
        return self


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    method: Literal["baseline", "metric"]
    ranking: pd.DataFrame
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("ranking", mode="before")
    @classmethod
    def _coerce_ranking(cls, v: Any) -> pd.DataFrame:
        if isinstance(v, pd.DataFrame):
            df = v.copy()
        elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            # Expect a sequence of row dicts
            df = pd.DataFrame(list(v))
        else:
            raise TypeError(
                "ranking must be a pandas.DataFrame or a sequence of row-mapping dicts"
            )

        missing = [c for c in REQUIRED_RANKING_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"ranking is missing required columns: {missing}")
        return df

    @field_serializer("ranking")
    def _serialize_ranking(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        # Serialize to list-of-dicts for stable round-trip
        return df.to_dict(orient="records")
