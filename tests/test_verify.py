"""verify decides what still needs checking from the report it finds."""

import hashlib
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
        "files": [{"path": "xnnpack/x-2k.pte", "sha256": hashlib.sha256(b"pte").hexdigest()}],
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


def test_an_accepted_gate_keeps_its_measured_verdict_and_records_the_reason(tmp_path, monkeypatch):
    from pipeline import gate, verify

    out = write(tmp_path, {})
    reference = tmp_path / "gate.json"
    reference.write_text("{}")
    monkeypatch.setattr(
        verify.hub,
        "fetch",
        lambda *a, **k: type("S", (), {"config": {}, "total_params": 1, "tokenizer_config": None})(),
    )
    monkeypatch.setattr(verify.hub, "download", lambda *a, **k: None)
    monkeypatch.setattr(verify.families, "architecture", lambda *a, **k: type("A", (), {"total_params": 1})())
    monkeypatch.setattr(verify.smoke, "run", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(gate, "measure", lambda *a, **k: {})
    monkeypatch.setattr(
        gate,
        "check",
        lambda *a, **k: {
            "passed": False,
            "problems": ["7 of 8"],
            "agreed": 7,
            "graded": 8,
            "fp32_mean": 0.66,
            "export_mean": 0.51,
        },
    )
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    with pytest.raises(verify.ExportError, match="decision gate failed"):
        verify.run(out, "xnnpack", tmp_path / "w1", gate_reference=reference)
    summary = verify.run(out, "xnnpack", tmp_path / "w2", gate_reference=reference, accept_gate="near a tie")
    assert summary["passed"]
    saved = json.loads((out / "xnnpack" / "export-report-2k.json").read_text())["gate"]
    assert saved["passed"] is False and saved["override"] == {"reason": "near a tie"}


def test_a_verified_report_is_verified_again_when_its_file_was_rebuilt(tmp_path, reaches_smoke):
    out = write(tmp_path, {"smoke": {"passed": True}, "gate": {"passed": True}})
    (out / "xnnpack" / "x-2k.pte").write_bytes(b"rebuilt")
    with pytest.raises(Ran):
        verify.run(out, "xnnpack", tmp_path / "work")
