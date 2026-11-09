"""verify decides what still needs checking from the report it finds."""

import json

import pytest

from pipeline import verify


class Ran(Exception):
    """The smoke test was reached: verify did not skip the report."""


def write(tmp_path, report):
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "xnnpack" / "x-2k.pte").write_bytes(b"pte")
    base = {
        "window": {"context": 2048},
        "files": [{"path": "xnnpack/x-2k.pte"}],
        "tokenizer": "tokenizer.json",
        "source": {"id": "org/x", "sha": "0" * 40, "variant": "instruct"},
    }
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps({**base, **report}))
    return out


@pytest.fixture
def reaches_smoke(monkeypatch):
    def fetch(*args, **kwargs):
        raise Ran

    monkeypatch.setattr(verify.hub, "fetch", fetch)


@pytest.mark.parametrize("gate", [None, {"passed": False, "problems": ["7 of 8"]}])
def test_a_smoke_tested_report_is_gated_again_when_its_gate_is_missing_or_failed(tmp_path, reaches_smoke, gate):
    report = {"smoke": {"passed": True}} | ({"gate": gate} if gate else {})
    out = write(tmp_path, report)
    reference = tmp_path / "gate.json"
    reference.write_text("{}")
    with pytest.raises(Ran):
        verify.run(out, "xnnpack", tmp_path / "work", gate_reference=reference)


def test_a_report_that_passed_both_is_left_alone(tmp_path, reaches_smoke):
    out = write(tmp_path, {"smoke": {"passed": True}, "gate": {"passed": True}})
    reference = tmp_path / "gate.json"
    reference.write_text("{}")
    assert verify.run(out, "xnnpack", tmp_path / "work", gate_reference=reference)["passed"]
    out_ungated = write(tmp_path / "b", {"smoke": {"passed": True}})
    assert verify.run(out_ungated, "xnnpack", tmp_path / "work")["passed"]
