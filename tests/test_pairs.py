from __future__ import annotations

import pandas as pd

from scperturb_cmap.data.pairs import PairingSpec, prepare_pair_table


def test_prepare_pair_table_with_negative_sampling():
    positives = pd.DataFrame(
        {
            "target_id": ["d1", "d1", "d2"],
            "signature_id": ["s1", "s2", "s3"],
        }
    )
    library_meta = pd.DataFrame(
        {
            "signature_id": ["s1", "s2", "s3", "s4", "s5"],
            "cell_line": ["CL1", "CL1", "CL2", "CL1", "CL2"],
        }
    )

    table = prepare_pair_table(
        positives,
        library_meta=library_meta,
        negatives_per_target=2,
        match_cell_line=True,
        random_state=0,
    )

    assert set(table.columns) == {"left_id", "right_id", "label"}
    assert (table["label"].isin({0, 1})).all()
    assert {"s1", "s2", "s3"}.issubset(set(table["right_id"]))
    negatives = table[table["label"] == 0]
    assert not negatives.empty


def test_prepare_pair_table_uses_existing_labels():
    positives = pd.DataFrame(
        {
            "target_id": ["t1", "t1"],
            "signature_id": ["s1", "s2"],
            "label": [1, 0],
        }
    )
    library_meta = pd.DataFrame(
        {
            "signature_id": ["s1", "s2"],
            "cell_line": ["CL1", "CL1"],
        }
    )

    table = prepare_pair_table(positives, library_meta=library_meta, spec=PairingSpec())
    assert len(table) == 2
    assert set(table["label"]) == {0, 1}
