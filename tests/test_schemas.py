import pandas as pd
import pytest

from scperturb_cmap.io.schemas import (
    DrugSignature,
    ScoreResult,
    TargetSignature,
)


def test_target_signature_validation_roundtrip():
    data = {
        "genes": ["GENE1", "GENE2"],
        "weights": [0.5, -1.25],
        "metadata": {"source": "unit-test"},
    }
    model = TargetSignature.model_validate(data)
    assert model.genes == data["genes"]
    assert model.weights == data["weights"]
    assert model.metadata == data["metadata"]
    # round-trip
    assert model.model_dump() == data


def test_target_signature_length_mismatch_raises():
    data = {
        "genes": ["A", "B"],
        "weights": [1.0],
        "metadata": {},
    }
    with pytest.raises(Exception):
        TargetSignature.model_validate(data)


def test_drug_signature_validation_roundtrip():
    data = {
        "signature_id": "sig-001",
        "compound": "CMPD42",
        "cell_line": "HEK293",
        "genes": ["G1", "G2", "G3"],
        "values": [0.1, 0.0, -2.3],
        "metadata": {"batch": 1},
    }
    model = DrugSignature.model_validate(data)
    assert model.signature_id == data["signature_id"]
    assert model.compound == data["compound"]
    assert model.cell_line == data["cell_line"]
    assert model.genes == data["genes"]
    assert model.values == data["values"]
    assert model.metadata == data["metadata"]
    # round-trip
    assert model.model_dump() == data


def test_score_result_validation_roundtrip():
    ranking_rows = [
        {
            "signature_id": "sig-001",
            "compound": "CMPD42",
            "cell_line": "HEK293",
            "score": 3.14,
            "moa": "inhibitor",
            "target": "MAPK1",
        },
        {
            "signature_id": "sig-002",
            "compound": "CMPD7",
            "cell_line": "A549",
            "score": -1.0,
            "moa": "agonist",
            "target": "EGFR",
        },
    ]
    data = {
        "method": "example",
        "ranking": ranking_rows,
        "metadata": {"n": 2},
    }

    model = ScoreResult.model_validate(data)
    # Internally a DataFrame
    assert isinstance(model.ranking, pd.DataFrame)
    assert list(model.ranking.columns) == [
        "signature_id",
        "compound",
        "cell_line",
        "score",
        "moa",
        "target",
    ]
    # round-trip back to list-of-dicts
    dumped = model.model_dump()
    assert dumped == data


def test_score_result_missing_required_column_raises():
    bad_rows = [
        {
            "signature_id": "sig-001",
            "compound": "CMPD42",
            # missing cell_line
            "score": 3.14,
            "moa": "inhibitor",
            "target": "MAPK1",
        }
    ]
    with pytest.raises(Exception):
        ScoreResult.model_validate({"method": "x", "ranking": bad_rows, "metadata": {}})
