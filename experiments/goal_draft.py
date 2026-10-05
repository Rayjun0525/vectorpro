"""Evaluate goal drafts independently; never approve or execute model proposals."""
import argparse
import json
from pathlib import Path
from vectorpro.goal_draft import propose_goal
from vectorpro.reference_evidence import ReferenceProviders
from experiments.reference_acquisition import manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/goal_draft_gemma"))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    args.output.mkdir(parents=True)
    common = [{"kind": "equals_initial", "parameter": "destination", "source": "source"},
              {"kind": "unchanged_except", "parameters": ["source", "destination"]}]
    cases = [{"name": "copy", "intent": "Duplicate source.bin into target.bin, preserving source.bin.",
              "expected_rules": common + [{"kind": "unchanged", "parameter": "source"}]},
             {"name": "move", "intent": "Rename source.bin to target.bin. Afterward source.bin must not exist.",
              "expected_rules": common + [{"kind": "absent", "parameter": "source"}]},
             {"name": "unsupported", "intent": "Encrypt source.bin into target.bin using a secret key.", "expected_rules": None}]
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    providers = ReferenceProviders(manifests())
    (args.output / "interfaces.json").write_text(json.dumps(providers.describe(), indent=2), encoding="utf-8")
    from experiments.verified_acquisition_gemma import GemmaModel
    model = GemmaModel("/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf")
    def canonical(rules):
        return sorted(json.dumps({**r, **({"parameters": sorted(r["parameters"])} if "parameters" in r else {})}, sort_keys=True) for r in rules)
    outcomes = []
    for case in cases:
        result = propose_goal(model, providers, case["intent"])
        passed = result["status"] == "needs_input" if case["expected_rules"] is None else (
            result["status"] == "needs_goal_review" and canonical(result["goal"]["rules"]) == canonical(case["expected_rules"]))
        outcomes.append({"name": case["name"], "passed": passed, "result": result})
        (args.output / "summary.json").write_text(json.dumps({"protocol": "goal-draft-only; no approval or execution", "cases": outcomes}, indent=2), encoding="utf-8")
        (args.output / "raw_calls.json").write_text(json.dumps(model.raw, indent=2), encoding="utf-8")
        print(json.dumps({"name": case["name"], "passed": passed, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
