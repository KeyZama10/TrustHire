"""Custom test-set schema: segment validation, window labelling, manifest validation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from trusthire.evaluation.dataset import SessionLabels, load_manifest, validate, write_template


def _labels(*segs: tuple[float, float, str]) -> SessionLabels:
    return SessionLabels.model_validate(
        {"session_id": "s", "segments": [{"start": a, "end": b, "label": c} for a, b, c in segs]}
    )


def test_window_label_majority_rule():
    lab = _labels((0, 10, "genuine"), (10, 30, "face_swap"))
    assert lab.label_at(0, 3) == (0, "genuine")
    assert lab.label_at(9.5, 12.5) == (1, "face_swap")  # 2.5 s of 3 s inside the attack
    assert lab.label_at(8.5, 11.5)[0] == 1  # exactly 50 % overlap -> fraud (rule is ">= 50 %")
    assert lab.label_at(8.0, 11.0)[0] == 0  # 1 s of 3 s (33 %) -> genuine


def test_overlapping_or_inverted_segments_rejected():
    with pytest.raises(ValidationError):
        _labels((0, 10, "genuine"), (9, 20, "proxy"))
    with pytest.raises(ValidationError):
        _labels((5, 1, "genuine"))
    with pytest.raises(ValidationError):
        _labels((0, 5, "not_an_attack"))


def test_template_and_validation(tmp_path):
    root = write_template(tmp_path / "custom")
    df = load_manifest(root)
    assert list(df.session_id) == ["s001"]
    problems = validate(root)
    assert any("missing recording" in p for p in problems)  # template has no media yet
    (root / "recordings" / "s001.mp4").write_bytes(b"x")
    for f in ("enroll/p01_id.jpg", "enroll/p01_voice.wav", "resumes/p01.pdf"):
        (root / f).write_bytes(b"x")
    assert validate(root) == []
    (root / "labels" / "s001.json").write_text(
        json.dumps({"session_id": "s001", "segments": [{"start": 0, "end": 60, "label": "genuine"}]})
    )
    assert any("never appears" in p for p in validate(root))
