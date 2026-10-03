"""M2: the learning loop. Plans in, vector programs out, no per-task code or structure.

A curriculum of plans (experiments/curricula/*.json) is fed to one learner
in order. The plans are plain data, standing in for what a language model
would produce from a requirement. For each plan the learner draws examples
round by round (8, 16, 32, 64 at 4 bits, plus 32 validation examples at 8
bits). It first tries to compose what the registry already holds, otherwise
learns a new unit by trying generic structures, and registers the result once
it reproduces every example.

The experimenter-only parts never feed back into learning:
  * TARGETS below stand in for human labels or traces of an existing system.
  * The probe measures each round's candidate on 16- and 32-bit inputs.
  * The audit compares learned unit tables with known local rules.

Usage: python experiments/learning_loop.py [--curriculum PATH] [--seeds ...]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from vectorpro.data import random_tuples
from vectorpro.execution import BitExecutable
from vectorpro.learning import ExampleSet, Learner, LearningOutcome, LearningPlan, Registry
from vectorpro.tasks import AND, FULL_ADDER, FULL_SUBTRACTOR, LESS_THAN, MUX, OR, XOR, LocalRule
from vectorpro.verification import verify_local_rule

from targets import TARGETS

HERE = Path(__file__).parent


AUDIT_RULES: dict[str, LocalRule] = {
    "and": AND.local_rule, "or": OR.local_rule, "xor": XOR.local_rule, "mux": MUX.local_rule,
    "add": FULL_ADDER, "sub": FULL_SUBTRACTOR, "lt": LESS_THAN,
}

PROBE_WIDTHS, PROBE_SAMPLES = (16, 32), 300


def make_probe(plan: LearningPlan, seed: int):
    rng = random.Random(f"probe-{plan.name}-{seed}")
    sets = []
    for width in PROBE_WIDTHS:
        ops = random_tuples(plan.arity, width, PROBE_SAMPLES, rng)
        sets.append(ExampleSet(width, ops, [TARGETS[plan.name](o, width) for o in ops]))

    def probe(executable: BitExecutable) -> dict:
        return {f"{s.width}bit": s.accuracy(executable) for s in sets}

    return probe


def audit(outcome: LearningOutcome) -> dict | None:
    cap = outcome.capability
    rule = AUDIT_RULES.get(outcome.plan.name)
    if cap is None or rule is None or cap.provenance["kind"] != "unit":
        return None
    if cap.executable.cell.signature != rule.signature:
        return {"rule": rule.name, "comparable": False}
    return {**verify_local_rule(cap.executable.cell, rule).to_dict(), "comparable": True}


def run_seed(plans: list[LearningPlan], seed: int) -> tuple[dict, Registry]:
    torch.manual_seed(seed)
    registry = Registry()
    learner = Learner(registry)
    probes = {p.name: make_probe(p, seed) for p in plans}
    outcomes = {}
    for plan in plans:
        outcome = learner.learn(plan, TARGETS[plan.name], random.Random(f"{plan.name}-{seed}"),
                                probes[plan.name])
        if outcome.capability is not None:
            outcome.capability.audit = audit(outcome)
        outcomes[plan.name] = {
            "learned": outcome.learned,
            "history": outcome.history,
            "audit": outcome.capability.audit if outcome.capability else None,
        }
        last = outcome.history[-1]
        print(f"  {plan.name:<10} {'learned' if outcome.learned else 'NOT learned':<11} "
              f"via {last['strategy']:<5} from {last['train_examples']:>2} examples  "
              f"probe {last.get('probe')}", flush=True)

    # Continual learning check: everything learned earlier still behaves the same.
    retention = {
        cap.name: probes[cap.name](cap.executable) == cap.history[-1]["probe"] for cap in registry
    }
    return {"seed": seed, "outcomes": outcomes, "retained": retention}, registry


def requests_demo(registry: Registry) -> dict:
    """How a user reaches learned capabilities: by words, by examples, by name."""
    rng = random.Random("requests")
    ops = [tuple(rng.getrandbits(8) for _ in range(3)) for _ in range(8)]
    unnamed = ExampleSet(8, ops, [(a + b + c) & 0xFF for a, b, c in ops])
    run_ops = [(1, 2, 3), (2**31, 2**31, 5), (2**32 - 1, 1, 0)]
    return {
        "search 'sum'": [c.name for c in registry.search("sum")],
        "search 'bitwise'": [c.name for c in registry.search("bitwise")],
        "find_by_examples (unnamed 3-input examples)": [c.name for c in registry.find_by_examples(3, unnamed)],
        "run add3 at 32 bits": {str(o): r for o, r in zip(run_ops, registry.run("add3", run_ops, 32))}
        if "add3" in registry else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curriculum", type=Path, default=HERE / "curricula" / "arithmetic.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--out", type=Path, default=Path("results/learning_loop.json"))
    parser.add_argument("--registry-out", type=Path, default=Path("results/registry.json"))
    args = parser.parse_args()

    plans = [LearningPlan.from_dict(p) for p in json.loads(args.curriculum.read_text())["plans"]]
    runs, first_registry = [], None
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        record, registry = run_seed(plans, seed)
        runs.append(record)
        first_registry = first_registry or registry

    first_registry.save(args.registry_out)
    reloaded = Registry.load(args.registry_out)
    demo = requests_demo(reloaded)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "learning_loop",
        "curriculum": [p.to_dict() for p in plans],
        "runs": runs,
        "requests_demo": demo,
    }, indent=1))

    print("\nsummary (learned seeds | strategy | examples used per seed)")
    for plan in plans:
        outs = [r["outcomes"][plan.name] for r in runs]
        learned = sum(o["learned"] for o in outs)
        strategies = sorted({o["history"][-1]["strategy"] for o in outs if o["learned"]}) or ["-"]
        examples = [o["history"][-1]["train_examples"] if o["learned"] else "x" for o in outs]
        print(f"  {plan.name:<10} {learned}/{len(runs)}  {','.join(strategies):<12} {examples}")
    print("retained after curriculum:", all(all(r["retained"].values()) for r in runs))
    print("\nrequests on the reloaded registry:")
    for k, v in demo.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {args.out} and {args.registry_out}")


if __name__ == "__main__":
    main()
