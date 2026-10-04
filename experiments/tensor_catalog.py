"""Prototype: lossless typed tensor tree + real model search vectors in ONE file.

Strings remain losslessly encoded UTF-8, not magically language-free semantics.
Qwen hidden-state mean pooling is experimental, not a trained retrieval model.
Existing runtime storage is unchanged; this tests a prospective storage boundary.
"""
import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from vectorpro.host import HostContext
from vectorpro.learning.registry import Registry
from vectorpro.runtime import VectorRuntime
from vectorpro.agent import AgentSession


def encode_tree(data):
    tape, blob, floats = [], bytearray(), []

    def visit(value):
        if value is None:
            tape.append([0, 0, 0])
        elif isinstance(value, bool):
            tape.append([1, int(value), 0])
        elif isinstance(value, int):
            tape.append([2, value, 0])
        elif isinstance(value, float):
            tape.append([3, len(floats), 0]); floats.append(value)
        elif isinstance(value, str):
            raw = value.encode("utf-8")
            tape.append([4, len(blob), len(raw)]); blob.extend(raw)
        elif isinstance(value, list):
            tape.append([5, len(value), 0])
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            tape.append([6, len(value), 0])
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("tree object keys must be strings")
                visit(key); visit(child)
        else:
            raise ValueError(f"unsupported node {type(value)}")
    visit(data)
    return {"nodes": torch.tensor(tape, dtype=torch.int64),
            "bytes": torch.tensor(list(blob), dtype=torch.uint8),
            "floats": torch.tensor(floats, dtype=torch.float64)}


def decode_tree(tree):
    tape = tree["nodes"].tolist()
    blob = bytes(tree["bytes"].tolist())
    floats = tree["floats"].tolist()
    cursor = 0

    def read():
        nonlocal cursor
        tag, value, length = tape[cursor]; cursor += 1
        if tag == 0: return None
        if tag == 1: return bool(value)
        if tag == 2: return value
        if tag == 3: return floats[value]
        if tag == 4: return blob[value:value + length].decode("utf-8")
        if tag == 5: return [read() for _ in range(value)]
        if tag == 6:
            result = {}
            for _ in range(value):
                key = read(); result[key] = read()
            return result
        raise ValueError("unknown tensor node type")
    result = read()
    if cursor != len(tape):
        raise ValueError("trailing tensor nodes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/initial_model/program.json"))
    parser.add_argument("--root", type=Path, default=Path("results/tensor_catalog"))
    parser.add_argument("--model", default="/opt/vectorpro-models/Qwen3-0.6B-Q8_0.gguf")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.source.read_text(encoding="utf-8"))
    # Exercise types beyond the current English-only catalog too.
    probe = {"한국어": [None, True, False, -42, 1.125, "경로/파일.bin", {}, []]}
    assert decode_tree(encode_tree(probe)) == probe
    exact = encode_tree(original)
    assert decode_tree(exact) == original
    caps = original["registry"]["capabilities"]
    names = [c["plan"]["name"] for c in caps]
    texts = [c["plan"]["name"] + ": " + c["plan"]["description"] for c in caps]
    from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
    model = Llama(model_path=args.model, embedding=True, pooling_type=LLAMA_POOLING_TYPE_MEAN,
                  n_ctx=2048, n_threads=4, n_batch=512, verbose=False)

    def embed(text):
        vector = model.create_embedding(text)["data"][0]["embedding"]
        return F.normalize(torch.tensor(vector, dtype=torch.float32), dim=0)

    vectors = torch.stack([embed(text) for text in texts])
    archive = {"version": torch.tensor([1]), **exact, "semantic_vectors": vectors,
               **{"embedding_" + k: v for k, v in encode_tree({
                   "model": "Qwen3-0.6B-Q8_0", "pooling": "mean", "trained_retrieval": False}).items()}}
    path = args.root / "program.pt"
    assert all(isinstance(v, torch.Tensor) for v in archive.values())
    torch.save(archive, path)
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    restored = decode_tree(loaded)
    assert restored == original
    assert torch.equal(loaded["semantic_vectors"], vectors)
    host = HostContext(args.root)
    runtime = VectorRuntime(Registry.from_data(restored["registry"]), host=host)
    rng = random.Random(91)
    numeric_ok = 0
    for _ in range(100):
        a, b = rng.getrandbits(16), rng.getrandbits(16)
        assert runtime.request("xor", [(a, b)], 16).outputs == [a ^ b]
        numeric_ok += 1
    (args.root / "input.bin").write_bytes(b"\x00\x07\xff\x80")
    agent = AgentSession(runtime, None)
    assert agent.call("execute", {"name": "map", "width": 16,
            "operands": [[{"utf8": "input.bin"}, 53]]})["outputs"] == [4]
    assert (args.root / "input.bin").read_bytes() == b"\x35\x32\xca\xb5"
    catalog = agent.call("list_capabilities", {})
    # JSON is reconstructed at the interchange boundary, not stored in the archive.
    (args.root / "exported-catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    queries = [("xor", "Compute the bitwise exclusive OR of two numbers"),
               ("sub", "Subtract one integer from another"),
               ("map", "Apply a byte XOR mask to every byte in a file"),
               ("fill", "Fill a file with a repeated byte value")]
    retrieval = []
    for expected, query in queries:
        scores = loaded["semantic_vectors"] @ embed(query)
        order = scores.argsort(descending=True).tolist()
        retrieval.append({"query": query, "expected": expected, "top1": names[order[0]],
                          "top3": [names[i] for i in order[:3]],
                          "top1_passed": names[order[0]] == expected,
                          "top3_passed": expected in [names[i] for i in order[:3]]})
    # Editing metadata then re-encoding must not silently alter execution tensors.
    restored["registry"]["capabilities"][0]["plan"]["tags"].append("왕복 검증")
    edited = decode_tree(encode_tree(restored))
    assert edited == restored
    assert edited["registry"]["capabilities"][0]["provenance"] == original["registry"]["capabilities"][0]["provenance"]
    # Persist the edit back into the SAME tensor file, updating its search vector too.
    loaded.update(encode_tree(edited))
    updated_plan = edited["registry"]["capabilities"][0]["plan"]
    loaded["semantic_vectors"][0] = embed(updated_plan["name"] + ": " + updated_plan["description"] +
                                          " " + " ".join(updated_plan["tags"]))
    torch.save(loaded, path)
    reread = torch.load(path, weights_only=True, map_location="cpu")
    assert decode_tree(reread) == edited
    assert torch.equal(reread["semantic_vectors"], loaded["semantic_vectors"])
    summary = {"exact_roundtrip": True, "unicode_types_roundtrip": True,
        "tensor_only_archive": True, "capabilities": len(caps), "numeric_heldout": numeric_ok,
        "native_file_passed": True, "metadata_edit_roundtrip": True,
        "semantic_top1": sum(r["top1_passed"] for r in retrieval),
        "semantic_top3": sum(r["top3_passed"] for r in retrieval),
        "semantic_cases": len(retrieval), "retrieval": retrieval,
        "archive_bytes": path.stat().st_size, "source_json_bytes": args.source.stat().st_size,
        "embedding": "Qwen causal hidden-state mean pooling, NOT a trained sentence encoder",
        "storage": "typed numeric tree + UTF-8 byte tensor + float64 values; language remains encoded",
        "prototype_only": True}
    (args.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
