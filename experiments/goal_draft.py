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
    parser.add_argument("--encoding", choices=("per_parameter", "states", "rules"), default="per_parameter")
    parser.add_argument("--suite", choices=("replay", "heldout"), default="replay")
    parser.add_argument("--choice-adapter", action="store_true")
    parser.add_argument("--compact-choice", action="store_true")
    parser.add_argument("--word-choice", action="store_true")
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--goal-memory", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    args.output.mkdir(parents=True)
    common = [{"kind": "equals_initial", "parameter": "destination", "source": "source"},
              {"kind": "unchanged_except", "parameters": ["source", "destination"]}]
    cases = [{"name": "copy", "intent": "Duplicate source.bin into target.bin, preserving source.bin.",
              "expected_rules": [common[0], {"kind": "unchanged_except", "parameters": ["destination"]}, {"kind": "unchanged", "parameter": "source"}]},
             {"name": "move", "intent": "Rename source.bin to target.bin. Afterward source.bin must not exist.",
              "expected_rules": common + [{"kind": "absent", "parameter": "source"}]},
             {"name": "unsupported", "intent": "Encrypt source.bin into target.bin using a secret key.", "expected_rules": None}]
    if args.suite == "heldout":
        cases = json.loads(Path("experiments/goal_draft_heldout.json").read_text(encoding="utf-8"))
    if args.cases_file:
        cases = json.loads(args.cases_file.read_text(encoding="utf-8"))
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    providers = ReferenceProviders(manifests())
    (args.output / "interfaces.json").write_text(json.dumps(providers.describe(), indent=2), encoding="utf-8")
    memory, model = None, None
    if args.goal_memory:
        from vectorpro.goal_memory import GoalMemory
        from vectorpro.semantic_catalog import Encoder
        from vectorpro.runtime import VectorRuntime
        directory = Path("/opt/vectorpro-models/multilingual-minilm")
        encoder, identity = Encoder(directory), json.loads((directory / "download.json").read_text())
        seeds = json.loads(Path("experiments/goal_memory_seed.json").read_text())
        (args.output / "teacher_examples.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
        memory = GoalMemory(seeds, encoder, identity)
        runtime = VectorRuntime.load("results/reference_acquisition_gemma/program.pt")
        memory.attach(runtime)
        runtime.save(args.output / "program.pt")
        memory = GoalMemory.from_runtime(VectorRuntime.load(args.output / "program.pt"), encoder, identity)
    else:
        from experiments.verified_acquisition_gemma import GemmaModel
        if args.choice_adapter:
            from experiments.goal_choice_model import GoalChoiceModel
            GemmaModel = GoalChoiceModel
        model = GemmaModel("/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf", compact=args.compact_choice, words=args.word_choice) if args.choice_adapter else GemmaModel("/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf")
    def canonical(rules):
        return sorted(json.dumps({**r, **({"parameters": sorted(r["parameters"])} if "parameters" in r else {})}, sort_keys=True) for r in rules)
    outcomes = []
    for case in cases:
        result = propose_goal(model, providers, case["intent"], encoding=args.encoding, goal_memory=memory)
        passed = result["status"] == "needs_input" if case["expected_rules"] is None else (
            result["status"] == "needs_goal_review" and canonical(result["goal"]["rules"]) == canonical(case["expected_rules"]))
        outcomes.append({"name": case["name"], "passed": passed, "result": result})
        (args.output / "summary.json").write_text(json.dumps({"protocol": "goal-draft-only; no approval or execution", "encoding": args.encoding, "suite": args.suite, "choice_adapter": args.choice_adapter, "compact_choice": args.compact_choice, "word_choice": args.word_choice, "goal_memory": args.goal_memory, "cases": outcomes}, indent=2), encoding="utf-8")
        (args.output / "raw_calls.json").write_text(json.dumps(model.raw if model else [], indent=2), encoding="utf-8")
        print(json.dumps({"name": case["name"], "passed": passed, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
