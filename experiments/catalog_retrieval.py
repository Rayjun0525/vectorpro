"""Frozen retrieval evaluation with a trained multilingual encoder and tensor-derived documents.

Search returns candidates only, never executes. Descriptions of host primitives
are machine semantics; learned task descriptions are reconstructed from routing.
No query-to-task dispatch or test queries are stored in the search index.
"""
import argparse
import json
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .tensor_catalog import encode_tree, decode_tree
except ImportError:
    from tensor_catalog import encode_tree, decode_tree
from vectorpro.host import HOST_TYPES, HostContext
from vectorpro.learning.registry import Registry
from vectorpro.machine import VectorProgram
from vectorpro.runtime import VectorRuntime
from vectorpro.agent import AgentSession


from vectorpro.semantic_catalog import document, Encoder, input_types

def evaluate(vectors, names, encoder, cases, registry=None):
    queries = encoder([c["query"] for c in cases])
    scores = queries @ vectors.T
    if registry is not None:
        for i, case in enumerate(cases):
            for j, name in enumerate(names):
                if input_types(registry.get(name)) != case["input_types"]:
                    scores[i, j] = float('-inf')
    order = scores.argsort(descending=True)
    rows = []
    for case, ranking in zip(cases, order.tolist()):
        ranked = [names[i] for i in ranking if torch.isfinite(scores[len(rows), i])][:3]
        rows.append({**case, "top1": ranked[0] if ranked else None, "top3": ranked,
                     "top1_passed": bool(ranked) and case["expected"] == ranked[0],
                     "top3_passed": case["expected"] in ranked})
    return {"top1": sum(r["top1_passed"] for r in rows),
            "top3": sum(r["top3_passed"] for r in rows), "cases": len(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/opt/vectorpro-models/multilingual-minilm")
    parser.add_argument("--root", type=Path, default=Path("results/catalog_retrieval"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path("results/initial_model/program.json").read_text())
    registry = Registry.from_data(data["registry"])
    names = [c.name for c in registry]
    plain = [c.plan.description for c in registry]
    derived = [document(registry, c.name) for c in registry]
    renamed = copy.deepcopy(data)
    for i, c in enumerate(renamed["registry"]["capabilities"]):
        c["plan"]["name"] = f"opaque_{i}"
    anonymous = Registry.from_data(renamed["registry"])
    assert derived == [document(anonymous, c.name) for c in anonymous]
    encoder = Encoder(args.model)
    plain_vectors = encoder(plain)
    vectors = encoder(derived)
    cases = json.loads(Path("experiments/requests/retrieval_evaluation.json").read_text(encoding="utf-8"))
    results = {kind: {split: evaluate(v, names, encoder, cases[split]) for split in cases}
               for kind, v in (("description_only", plain_vectors), ("tensor_derived", vectors))}
    results["typed_tensor_derived"] = {split: evaluate(vectors, names, encoder, split_cases, registry)
        for split, split_cases in cases.items() if all("input_types" in c for c in split_cases)}
    archive = {"version": torch.tensor([1]), **encode_tree(data), "semantic_vectors": vectors,
               **{"catalog_" + k: v for k, v in encode_tree({
                   "names": names, "documents": derived,
                   "contracts": [{"name": c.name, "input_types": input_types(c),
                                  "output_width": c.plan.output.value,
                                  "has_effects": bool(c.executable.effects)} for c in registry],
                   "encoder": json.loads((Path(args.model) / "download.json").read_text()),
                   "pooling": "masked_mean", "max_length": 128}).items()}}
    torch.save(archive, args.root / "program.pt")
    loaded = torch.load(args.root / "program.pt", weights_only=True)
    assert decode_tree(loaded) == data
    assert torch.equal(loaded["semantic_vectors"], vectors)
    restored_catalog = decode_tree({k: loaded["catalog_" + k] for k in ("nodes", "bytes", "floats")})
    assert restored_catalog["documents"] == derived
    assert [c["input_types"] for c in restored_catalog["contracts"]] == [input_types(c) for c in registry]
    host = HostContext(args.root)
    runtime = VectorRuntime(Registry.from_data(decode_tree(loaded)["registry"]), host=host)
    # Actually execute the top-ranked development numeric/file candidates too.
    chosen = results["tensor_derived"]["development"]["rows"]
    executed = []
    for row in chosen:
        expected = row["expected"]
        if row["top1"] != expected:
            executed.append({"expected": expected, "passed": False, "reason": "wrong candidate"})
            continue
        if expected in ("map", "fill"):
            (args.root / "input.bin").write_bytes(b"\x00\x07\xff\x80")
            result = AgentSession(runtime, None).call("execute", {"name": row["top1"], "width": 16,
                     "operands": [[{"utf8": "input.bin"}, 53]]})
            desired = b"\x35\x32\xca\xb5" if expected == "map" else b"\x35" * 4
            passed = result["outputs"] == [4] and (args.root / "input.bin").read_bytes() == desired
        else:
            output = runtime.request(row["top1"], [(12345, 4567)], 16).outputs
            desired = 12345 ^ 4567 if expected == "xor" else 12345 - 4567
            passed = output == [desired]
        executed.append({"expected": expected, "passed": passed})
    summary = {"encoder": json.loads((Path(args.model) / "download.json").read_text()),
               "model_parameters": sum(p.numel() for p in encoder.model.parameters()),
               "exact_roundtrip": True, "search_then_execution": executed,
               "results": results, "documents": dict(zip(names, derived)),
               "queries_used_to_train_encoder": False, "core_runtime_changed": False}
    summary["task_names_not_used_to_derive_documents"] = True
    summary["typed_constraints_source"] = "explicit caller-supplied input_types; never inferred from expected label"
    summary["catalog_contracts_roundtrip"] = True
    summary["evaluation_protocol"] = {
        "development": "original four requests; used while improving extraction",
        "heldout": "24 diagnostic requests inspected during iteration, no longer independent holdout",
        "final_holdout": "24 new English/Korean requests frozen before final comparison; not used to tune retrieval",
        "scope": "12-capability prototype; types are explicit caller context"}
    summary["typed_ambiguity"] = {
        split: {"multiple_candidate_cases": sum(sum(input_types(registry.get(n)) == c["input_types"] for n in names) > 1 for c in cases[split]),
                "single_candidate_cases": sum(sum(input_types(registry.get(n)) == c["input_types"] for n in names) == 1 for c in cases[split])}
        for split in results["typed_tensor_derived"]}
    (args.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {s: {m: r[m] for m in ('top1', 'top3', 'cases')} for s, r in splits.items()}
                      for k, splits in results.items()}), flush=True)
    print(json.dumps({"search_then_execution": executed}), flush=True)


if __name__ == "__main__":
    main()
