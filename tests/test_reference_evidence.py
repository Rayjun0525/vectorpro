import copy
import json
import subprocess
from pathlib import Path
import pytest
from vectorpro.acquisition import EvidenceBank
from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.reference_evidence import ReferenceProviders, snapshot
from vectorpro.runtime import VectorRuntime
from experiments.reference_acquisition import manifests, ProtocolModel


@pytest.mark.parametrize("provider", ["copy", "move"])
def test_collection_learning_save_and_reuse_are_separate(provider, tmp_path, monkeypatch):
    (tmp_path / "input.bin").write_bytes(b"new\xff")
    (tmp_path / "keep").write_bytes(b"keep")
    (tmp_path / "empty").mkdir()
    original = subprocess.Popen
    roots = []
    def watch(*args, **kwargs):
        roots.append(Path(kwargs["cwd"]))
        assert Path(kwargs["cwd"]) != tmp_path
        return original(*args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", watch)
    runtime = VectorRuntime(host=HostContext(tmp_path))
    model = ProtocolModel(provider)
    complete = model.complete
    def phased(messages, tools):
        if model.calls == 1:
            assert (tmp_path / "input.bin").read_bytes() == b"new\xff"
            assert not runtime.host.events
            def forbidden(*args, **kwargs):
                pytest.fail("learner launched the external reference")
            monkeypatch.setattr(subprocess, "Popen", forbidden)
        return complete(messages, tools)
    model.complete = phased
    program = tmp_path / "program.pt"
    session = AgentSession(runtime, model, program, reference_providers=ReferenceProviders(manifests()))
    intent = provider + " input.bin to output.bin"
    result = session.run(intent)
    assert result["status"] == "executed"
    assert len(roots) == 7 and all(not root.exists() for root in roots)
    assert (tmp_path / "output.bin").read_bytes() == b"new\xff"
    assert (tmp_path / "input.bin").exists() is (provider == "copy")
    assert (tmp_path / "keep").read_bytes() == b"keep" and (tmp_path / "empty").is_dir()
    assert [e["name"] for e in result["tools"]] == ["collect_evidence", "learn_verified_contract", "call_contract"]
    assert session.collected_evidence["heldout"] and "executable sha256=" in session.collected_evidence["origin"]
    loaded = VectorRuntime.load(program, host=HostContext(tmp_path))
    (tmp_path / "input.bin").write_bytes(b"second")
    known = loaded.contract("observed_" + provider)
    reused = AgentSession(loaded, ProtocolModel(provider, known), allow_learning=False,
        reference_providers=ReferenceProviders(manifests())).run(intent)
    assert reused["status"] == "executed" and len(reused["tools"]) == 1


@pytest.mark.parametrize("bad", ["placeholder", "escape", "code", "type"])
def test_bad_reference_manifest_never_runs(bad):
    item = copy.deepcopy(manifests()[0])
    if bad == "placeholder":
        item["argv"].append("$unknown")
    elif bad == "escape":
        item["seed_files"]["../outside"] = "00"
    elif bad == "code":
        item["python"] = "copy(source,destination)"
    else:
        item["interface"]["parameters"][0]["type"] = "buffer"
    with pytest.raises(ValueError):
        ReferenceProviders([item])


def test_failed_reference_cannot_register_and_collection_not_repeated(tmp_path):
    item = copy.deepcopy(manifests()[0])
    item["argv"] = ["/bin/false"]
    session = AgentSession(VectorRuntime(host=HostContext(tmp_path)), None,
        reference_providers=ReferenceProviders([item]))
    session._intent = "copy input.bin to output.bin"
    with pytest.raises(ValueError, match="reference failed"):
        session.call("collect_evidence", {"provider_id": "copy"})
    assert not session.runtime.contracts() and not list(tmp_path.iterdir())
    assert "collect_evidence" not in [t["function"]["name"] for t in session.available_tools()]


def test_model_cannot_inject_argv_or_observation_answers(tmp_path):
    session = AgentSession(VectorRuntime(host=HostContext(tmp_path)), None,
        reference_providers=ReferenceProviders(manifests()))
    session._intent = "request"
    with pytest.raises(ValueError):
        session.call("collect_evidence", {"provider_id": "copy", "argv": ["/bin/true"]})
    with pytest.raises(ValueError):
        session.call("collect_evidence", {"provider_id": "made-up"})
    assert not session.runtime.contracts() and not list(tmp_path.iterdir())


def test_snapshot_rejects_links_extra_entries_and_excess_bytes(tmp_path):
    (tmp_path / "data").write_bytes(b"x" * 65537)
    with pytest.raises(ValueError, match="byte limit"):
        snapshot(tmp_path)
    (tmp_path / "data").unlink()
    (tmp_path / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        snapshot(tmp_path)


def test_reference_identity_is_checked_before_launch(tmp_path):
    executable = tmp_path / "reference"
    executable.write_bytes(Path("/bin/true").read_bytes())
    executable.chmod(0o755)
    item = copy.deepcopy(manifests()[0])
    item["argv"] = [str(executable)]
    providers = ReferenceProviders([item])
    executable.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        providers.collect("copy", "request")
