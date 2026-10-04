import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime
from vectorpro.semantic_catalog import TensorCatalog, operand_types, argument_roles
from vectorpro.tensor_codec import encode_tree, decode_tree


SOURCE = Path(__file__).resolve().parents[1] / "results/initial_model/program.json"


def uniform_encoder(texts):
    # Intentionally gives all functions equal similarity: evidence must disambiguate.
    return torch.ones((len(texts), 4)) / 2


def setup(tmp_path):
    runtime = VectorRuntime.load(SOURCE, host=HostContext(tmp_path))
    catalog = TensorCatalog(runtime, uniform_encoder, {"test": "equal scores"})
    return runtime, catalog, AgentSession(runtime, None, tmp_path / "program.pt", catalog=catalog)


def numeric(targets=(2, 6)):
    return {"query": "bitwise XOR", "operands": [[12345, 4567]], "width": 16,
            "numeric_validation": {"width": 4, "operands": [[1, 3], [5, 3]], "targets": list(targets)}}


def file_request(after="0106", keep="aa"):
    return {"query": "transform bytes of a file with XOR", "width": 16,
            "operands": [[{"utf8": "native.bin"}, 53]],
            "state_validation": [{"inputs": [{"utf8": "probe.bin"}, 1],
               "before": {"probe.bin": "0007", "keep.bin": "aa"},
               "after": {"probe.bin": after, "keep.bin": keep}, "output": 2}]}


def test_tensor_roundtrip_preserves_index_and_runs_without_llm(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    session.save()
    restored = VectorRuntime.load(tmp_path / "program.pt")
    assert restored.registry.to_data() == runtime.registry.to_data()
    assert torch.equal(restored._tensor_extras["semantic_vectors"], catalog.vectors)
    assert restored.request("xor", [(12345, 4567)], 16).outputs == [8686]
    restored.save(tmp_path / "second.pt")
    assert VectorRuntime.load(tmp_path / "second.pt").registry.to_data() == runtime.registry.to_data()
    assert torch.equal(VectorRuntime.load(tmp_path / "second.pt")._tensor_extras["semantic_vectors"], catalog.vectors)


def test_numeric_evidence_selects_despite_equal_scores_and_request_is_bound(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    assert session.call("resolve_request", numeric())["name"] == "xor"
    with pytest.raises(ValueError, match="exact name"):
        session.call("execute", {"name": "sub", "width": 16, "operands": [[12345, 4567]]})
    with pytest.raises(ValueError, match="exact name"):
        session.call("execute", {"name": "xor", "width": 16, "operands": [[1, 3]]})
    assert session.call("execute", {"name": "xor", "width": 16, "operands": [[12345, 4567]]})["outputs"] == [8686]
    with pytest.raises(ValueError, match="verify"):
        session.call("execute", {"name": "xor", "width": 16, "operands": [[12345, 4567]]})


def test_same_signature_unsupported_operation_is_not_nearest_substitution(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    args = numeric((3, 15))  # multiplication, neither stored subtraction nor XOR
    args["query"] = "multiply two numbers"
    result = session.call("resolve_request", args)
    assert result["status"] == "needs_learning_examples"
    assert result["matching_candidates"] == []
    with pytest.raises(ValueError, match="verify"):
        session.call("execute", {"name": "xor", "width": 16, "operands": [[12345, 4567]]})


def test_no_evidence_and_invalid_examples_cannot_authorize_execution(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    args = numeric(); args.pop("numeric_validation")
    assert session.call("resolve_request", args)["status"] == "needs_evidence"
    bad = numeric(); bad["numeric_validation"]["operands"] = [[1, 3], [1, 3]]
    with pytest.raises(ValueError, match="distinct"):
        session.call("resolve_request", bad)
    assert session._resolved is None
    with pytest.raises(ValueError, match="typed values"):
        operand_types([["file.bin", "53"]])


def test_examples_that_match_multiple_capabilities_require_more_evidence(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    args = numeric((1, 2))
    args["numeric_validation"]["operands"] = [[1, 0], [2, 0]]
    result = session.call("resolve_request", args)
    assert result["status"] == "needs_evidence"
    assert set(result["matching_candidates"]) == {"xor", "sub"}
    assert session._resolved is None


def test_teaching_refreshes_tensor_index_and_saved_function_runs_after_reload(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    request = json.loads((SOURCE.parents[2] / "experiments/requests/intercept_nand_teach.json").read_text())["request"]
    assert session.call("teach", {"request": request})["status"] == "learned"
    restored = VectorRuntime.load(tmp_path / "program.pt")
    assert "assistant_nand" in restored.registry
    assert len(restored._tensor_extras["semantic_vectors"]) == len(restored.registry.to_data()["capabilities"])
    assert restored.request("assistant_nand", [(12345, 4567)], 16).outputs == [61422]
    args = numeric((14, 14)); args["query"] = "bitwise complement of AND"
    assert session.call("resolve_request", args)["name"] == "assistant_nand"


def test_file_verification_uses_only_memory_and_compares_entire_snapshot(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    native = tmp_path / "native.bin"; native.write_bytes(b"\x00\x07\xff\x80")
    prior = native.read_bytes()
    assert session.call("resolve_request", file_request())["name"] == "map"
    assert native.read_bytes() == prior and runtime.host.events == []
    assert not (tmp_path / "probe.bin").exists()
    assert session.call("execute", {"name": "map", "width": 16,
               "operands": [[{"utf8": "native.bin"}, 53]]})["outputs"] == [4]
    assert native.read_bytes() == b"\x35\x32\xca\xb5"
    assert session.call("resolve_request", file_request(keep="bb"))["status"] == "needs_learning_examples"


def test_tensor_rejects_corrupt_versions_and_metadata_without_changing_file(tmp_path):
    runtime, catalog, session = setup(tmp_path); session.save()
    path = tmp_path / "program.pt"
    archive = torch.load(path, weights_only=True)
    archive["version"] = torch.tensor([99]); torch.save(archive, path)
    with pytest.raises(ValueError, match="version"):
        VectorRuntime.load(path)
    data = {"한국어": [True, None, 1.25, {}, [], 2**100, -(2**100), 2**63 - 1, -(2**63)]}
    valid = encode_tree(data)
    assert decode_tree(valid) == data
    bad = copy.deepcopy(valid); bad["nodes"][1, 1] = -1
    with pytest.raises(ValueError, match="byte range"):
        decode_tree(bad)


def test_llm_loop_extracts_operands_then_exposes_only_verified_execution(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    def message(name, args):
        return {"role": "assistant", "tool_calls": [{"id": "id", "function": {
            "name": name, "arguments": json.dumps(args)}}]}
    class Model:
        def __init__(self): self.turn = 0
        def complete(self, messages, tools):
            names = [t["function"]["name"] for t in tools]
            self.turn += 1
            if self.turn == 1:
                assert "execute_resolved" not in names
                return message("search_goal", {"query": "XOR"})
            if self.turn == 2:
                assert messages[-1]["content"] == "Compute the XOR of 12345 and 4567 at width 16"
                chosen = next(t for t in tools if t["function"]["description"].startswith("Prepare xor:"))
                index = int(chosen["function"]["name"].split("_")[1])
                return message(f"prepare_{index}", {"width": 16, "x0": 12345, "x1": 4567})
            if self.turn == 3:
                assert "12345" not in json.dumps(messages) and "4567" not in json.dumps(messages)
                assert "actual_input_0" in messages[-1]["content"]
                assert "check_small" in names and "execute_resolved" not in names
                schema = next(t for t in tools if t["function"]["name"] == "check_small")["function"]["parameters"]
                assert schema["properties"]["cases"]["items"]["properties"]["x0"]["maximum"] == 15
                return message("check_small", {"cases": [{"x0": 1, "x1": 3, "y": 2}, {"x0": 5, "x1": 3, "y": 6}]})
            if self.turn == 4:
                assert "execute_resolved" in names
                return message("execute_resolved", {})
            raise AssertionError("must stop after execution without another model turn")
    session.model = Model()
    result = session.run("Compute the XOR of 12345 and 4567 at width 16")
    assert result["status"] == "executed" and session.model.turn == 4
    assert result["tools"][-1]["result"]["outputs"] == [8686]


def test_model_text_cannot_claim_execution_without_tools(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    class Model:
        def complete(self, messages, tools):
            return {"role": "assistant", "content": "Successfully modified the file"}
    session.model = Model()
    assert session.run("Modify the file")["status"] == "not_executed"


def test_evidence_arrives_after_request_and_cannot_replace_requested_mask(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    session.max_calls = 3  # execution on the final permitted turn still completes
    (tmp_path / "native.bin").write_bytes(b"\x00\x07\xff\x80")
    candidate = next(c for c in catalog.contracts if c["name"] == "map")
    catalog.search_goal = lambda query, limit=5, types=None: [{**candidate, "description": "Map file bytes", "score": 1.0}]
    def message(name, args):
        return {"role": "assistant", "tool_calls": [{"id": "id", "function": {
            "name": name, "arguments": json.dumps(args)}}]}
    class Model:
        def __init__(self): self.turn = 0
        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                assert "probe.bin" not in json.dumps(messages)
                return message("search_goal", {"query": "mask file", "width": 16})
            if self.turn == 2:
                assert "probe.bin" not in json.dumps(messages)
                return message("prepare_0", {"width": 16, "x0": "native.bin", "x1": 53})
            if self.turn == 3:
                result = json.loads(messages[-1]["content"])
                assert result["status"] == "matched" and result["verification_source"] == "caller"
                assert result["request"]["operands"] == [[{"utf8": "native.bin"}, 53]]
                assert [t["function"]["name"] for t in tools] == ["execute_resolved"]
                return message("execute_resolved", {})
            raise AssertionError("must not request a model summary or second write")
    session.model = Model()
    response = session.run("Mask native.bin with 53", evidence=json.dumps(file_request()["state_validation"]))
    assert response["status"] == "executed" and session.model.turn == 3
    assert response["request"]["width"] == 16
    assert (tmp_path / "native.bin").read_bytes() == b"\x35\x32\xca\xb5"


def test_automatic_caller_verification_cannot_authorize_ambiguous_evidence(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    evidence = {"width":4,"operands":[[0,0],[1,0]],"targets":[0,1]}
    class Model:
        def __init__(self): self.turn = 0
        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                name, args = "search_goal", {"query":"numeric request"}
            elif self.turn == 2:
                assert "targets" not in json.dumps(messages)
                name, args = "prepare_0", {"width":8,"x0":7,"x1":3}
            elif self.turn == 3:
                assert json.loads(messages[-1]["content"])["status"] == "needs_evidence"
                assert "execute_resolved" not in [t["function"]["name"] for t in tools]
                name, args = "execute_resolved", {}
            else:
                assert json.loads(messages[-1]["content"])["status"] == "error"
                name, args = "ask_user", {"question":"Please provide distinguishing examples"}
            return {"role":"assistant","tool_calls":[{"id":"test","function":{"name":name,"arguments":json.dumps(args)}}]}
    session.model = Model()
    response = session.run("Compute with 7 and 3 at width 8", evidence=json.dumps(evidence))
    assert response["status"] == "needs_input" and session._resolved is None
    assert not any(e["result"].get("status") == "executed" for e in response["tools"])
    assert runtime.host.events == []


def test_model_invented_path_is_rejected_before_any_native_effect(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    path = tmp_path / '출력 폴더' / '한글 문서.bin'
    path.parent.mkdir(); path.write_bytes(b'abc')
    (tmp_path / 'IN').write_bytes(b'untouched')
    proof = {"cases":[{"inputs":[{"utf8":"probe.bin"},4],"before":{"probe.bin":"0010"},"after":{"probe.bin":"0404"},"output":2}]}
    class Model:
        def __init__(self): self.turn = 0
        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                name,args = 'search_goal',{'query':'fill bytes','width':16}
            elif self.turn in (2,3):
                prepare = next(t for t in tools if t['function']['name'].startswith('prepare_'))
                assert prepare['function']['parameters']['properties']['x0']['enum'] == ['출력 폴더/한글 문서.bin']
                assert 'probe.bin' not in json.dumps(messages)
                if self.turn == 3:
                    assert runtime.host.events == [] and path.read_bytes() == b'abc'
                    assert json.loads(messages[-1]['content'])['status'] == 'error'
                name,args = prepare['function']['name'], {'width':16,'x0':'IN' if self.turn==2 else '출력 폴더/한글 문서.bin','x1':23}
            else:
                name,args = 'execute_resolved',{}
            return {'role':'assistant','tool_calls':[{'id':'test','function':{'name':name,'arguments':json.dumps(args)}}]}
    session.model = Model()
    result = session.run('"출력 폴더/한글 문서.bin"을 23으로 채워줘. 폭은 16비트.',evidence=json.dumps(proof))
    assert result['status'] == 'executed' and result['outputs'] == [3]
    assert path.read_bytes() == bytes([23])*3 and (tmp_path/'IN').read_bytes() == b'untouched'


def test_execution_only_sessions_reject_teaching_even_if_model_hallucinates_tool(tmp_path):
    runtime = VectorRuntime()
    session = AgentSession(runtime, None, allow_learning=False)
    assert not any(t["function"]["name"].startswith("teach") for t in session.available_tools())
    with pytest.raises(ValueError, match="disabled"):
        session.call("teach", {"request": {"name": "bad"}})
    assert list(runtime.registry) == []


def test_argument_roles_come_from_cell_semantics_and_required_inputs(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    cap = copy.deepcopy(runtime.registry.get("sub"))
    cap.plan = replace(cap.plan, name="opaque_function")
    assert argument_roles(cap) == argument_roles(runtime.registry.get("sub"))
    assert "FROM" in argument_roles(cap)[0]
    session.call("search_goal", {"query": "subtract"})
    index = next(i for i, c in enumerate(session._candidates) if c["name"] == "sub")
    schema = next(t for t in session.available_tools() if t["function"]["name"] == f"prepare_{index}")["function"]["parameters"]
    assert schema["required"] == ["width", "x0", "x1"]
    with pytest.raises(ValueError, match="every candidate input"):
        session.call(f"prepare_{index}", {"width": 16, "x0": 12345})
    assert session._resolved is None and session._proposed is None


@pytest.mark.parametrize("state", [False, True])
def test_supplied_evidence_rejects_unsupported_goal_before_request_binding(tmp_path, state):
    runtime, catalog, session = setup(tmp_path)
    session.allow_learning = False
    (tmp_path / "native.bin").write_bytes(b"unchanged")
    evidence = {"cases": [{"inputs": [{"utf8": "probe.bin"}],
                           "before": {"probe.bin": "00"}, "after": {}}]} if state else {
                               "width": 4, "operands": [[2, 3], [4, 2]], "targets": [6, 8]}
    class Model:
        def complete(self, messages, tools):
            if len(messages) == 2:
                assert [t["function"]["name"] for t in tools] == ["search_goal"]
                args = {"query": "unsupported operation", **({"width": 16} if state else {})}
                name = "search_goal"
            else:
                result = json.loads(messages[-1]["content"])
                assert result["status"] == "needs_learning_examples"
                assert "verification" in result
                assert not any(t["function"]["name"].startswith("prepare_") for t in tools)
                name, args = "ask_user", {"question": "Please supply learning evidence"}
            return {"role": "assistant", "tool_calls": [{"id": "test", "function": {
                "name": name, "arguments": json.dumps(args)}}]}
    session.model = Model()
    response = session.run("Perform the unsupported operation", evidence=json.dumps(evidence))
    assert response["status"] == "needs_input"
    assert session._resolved is None and session._proposed is None
    assert runtime.host.events == [] and (tmp_path / "native.bin").read_bytes() == b"unchanged"


def test_precheck_does_not_bind_caller_demonstration_as_native_request(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    session._caller_evidence = {"numeric_validation": numeric()["numeric_validation"]}
    session._caller_types = ["value", "value"]
    result = session.call("search_goal", {"query": "xor"})
    assert [c["name"] for c in result["candidates"]] == ["xor"]
    assert session._resolved is None
    with pytest.raises(ValueError, match="uniquely matched"):
        session.call("execute_resolved", {})


def test_compact_examples_reject_large_or_duplicate_inputs(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    session.call("propose_numeric_request", {"query": "xor", "width": 16, "inputs": [12345, 4567]})
    with pytest.raises(ValueError, match="0..15"):
        session.call("check_small", {"cases": [{"x0": 12345, "x1": 4567, "y": 2}, {"x0": 5, "x1": 3, "y": 6}]})
    with pytest.raises(ValueError, match="distinct"):
        session.call("check_small", {"cases": [{"x0": 1, "x1": 3, "y": 2}] * 2})
    assert session._resolved is None


def small_nand():
    return {"name": "compact_nand", "output": "W", "training": [
        {"x0": 0, "x1": 0, "y": 15}, {"x0": 1, "x1": 1, "y": 14},
        {"x0": 1, "x1": 0, "y": 15}, {"x0": 0, "x1": 1, "y": 15}], "validation": [
        {"x0": 2, "x1": 1, "y": 15}, {"x0": 3, "x1": 1, "y": 14}]}


def test_compact_teaching_uses_disjoint_data_and_executes_original_inputs(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    session.call("propose_numeric_request", {"query": "NAND", "width": 8, "inputs": [6, 3]})
    assert session.call("check_small", {"cases": small_nand()["training"][:2]})["status"] == "needs_learning_examples"
    bad = small_nand(); bad["validation"][0] = bad["training"][0]
    with pytest.raises(ValueError, match="disjoint"):
        session.call("teach_small", bad)
    assert "compact_nand" not in runtime.registry
    result = session.call("teach_small", small_nand())
    assert result["status"] == "learned" and result["request_verification"]["status"] == "matched"
    assert session.call("execute_resolved", {})["outputs"] == [253]
    restored = VectorRuntime.load(tmp_path / "program.pt")
    import random
    rng = random.Random(81)
    inputs = [(rng.randrange(65536), rng.randrange(65536)) for _ in range(100)]
    assert restored.request("compact_nand", inputs, 16).outputs == [(~(a & b)) & 65535 for a, b in inputs]
    session.allow_learning = False
    with pytest.raises(ValueError, match="disabled"):
        session.call("teach_small", small_nand())


def test_model_invented_noop_snapshots_cannot_authorize_ambiguous_file_execution(tmp_path):
    runtime, catalog, session = setup(tmp_path)
    path = tmp_path / "native.bin"; path.write_bytes(b"unchanged")
    candidate = next(c for c in catalog.contracts if c["name"] == "guarded_map")
    catalog.search_goal = lambda query, limit=5, types=None: [{**candidate, "description": "Guarded file map", "score": 1.0}]
    class Model:
        def __init__(self): self.turn = 0
        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                name, args = "search_goal", {"query": "process the file"}
            elif self.turn == 2:
                name, args = "prepare_0", {"width": 16, "x0": "native.bin", "x1": 0, "x2": 0}
            elif self.turn == 3:
                assert [t["function"]["name"] for t in tools] == ["ask_user"]
                name, args = "verify_state", {"cases": [{"inputs": [{"utf8": "native.bin"}, 0, 0], "before": {}, "after": {}}]}
            else:
                assert json.loads(messages[-1]["content"])["status"] == "error"
                name, args = "ask_user", {"question": "What transformation and expected file state do you want?"}
            return {"role": "assistant", "tool_calls": [{"id": "test", "function": {"name": name, "arguments": json.dumps(args)}}]}
    session.model = Model()
    response = session.run("Process native.bin; I have not decided what I want")
    assert response["status"] == "needs_input"
    assert session._resolved is None and runtime.host.events == [] and path.read_bytes() == b"unchanged"
