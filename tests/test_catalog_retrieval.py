import json
from pathlib import Path

import torch

from experiments.catalog_retrieval import evaluate, input_types, document
from vectorpro.learning.registry import Registry


def registry():
    data = json.loads((Path(__file__).resolve().parents[1] / "results/initial_model/program.json").read_text())
    return Registry.from_data(data["registry"])


def test_typed_search_excludes_semantically_similar_wrong_signature_and_unknown_shape():
    catalog = registry()
    names = ["xor", "map", "pair_map"]
    # Deliberately make the wrong-signature functions more similar to the query.
    vectors = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.99, 0.01]])
    encode = lambda texts: torch.tensor([[1.0, 0.0] for _ in texts])
    cases = [{"query": "transform", "expected": "map", "input_types": ["path", "value"]},
             {"query": "transform", "expected": None, "input_types": ["buffer", "buffer", "buffer"]}]
    results = evaluate(vectors, names, encode, cases, catalog)
    assert results["rows"][0]["top3"] == ["map"]
    assert results["rows"][1]["top1"] is None
    assert results["rows"][1]["top3"] == []
    # The expected label must never change retrieval decisions.
    cases[0]["expected"] = "xor"
    assert evaluate(vectors, names, encode, cases, catalog)["rows"][0]["top1"] == "map"


def test_tensor_routing_distinguishes_fill_xor_condition_and_multiple_paths():
    catalog = registry()
    fill = document(catalog, "fill")
    mapping = document(catalog, "map")
    assert "same supplied value" in fill and "exclusive OR" not in fill
    assert "exclusive OR" in mapping
    assert "Conditionally" in document(catalog, "guarded_map")
    assert "two separate files" in document(catalog, "pair_map")
    assert input_types(catalog.get("file.read")) == ["path"]
