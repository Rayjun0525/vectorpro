"""M2c: learning loop and branch programs from examples. Nothing is authored.

One learner works through experiments/curricula/loops.json. For every plan
it draws examples round by round and tries, cheapest first: a composition of
registered capabilities, a new unit, then a loop (the body of a generic bit
fold). Every result is a vector program or a unit table. The plans hold no
structure, and no program is written by hand.

Controls, as in M2b: accuracy at 8/16/32 bits; every single re-routing of the
*learned* mul program must change its behaviour; the kernel sources name no
capability; the saved registry reloads identically.

Usage: python experiments/loop_learning.py [--seeds ...]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from targets import TARGETS
from vector_programs import accuracy, eval_sets, kernel_is_task_agnostic, rerouting_control
from vectorpro.data import random_tuples
from vectorpro.learning import Learner, LearningPlan, Registry

HERE = Path(__file__).parent


def run_seed(plans: list[LearningPlan], seed: int, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    registry = Registry(seed=seed)
    learner = Learner(registry)
    record: dict = {"seed": seed, "capabilities": {}}
    for plan in plans:
        outcome = learner.learn(plan, TARGETS[plan.name], random.Random(f"{plan.name}-{seed}"))
        last = outcome.history[-1]
        entry = {
            "learned": outcome.learned,
            "strategy": last["strategy"],
            "structure": last["structure"],
            "examples": last["train_examples"],
            "rounds": len(outcome.history),
            "validation_by_round": [h["validation_accuracy"] for h in outcome.history],
        }
        if outcome.learned:
            entry["accuracy"] = accuracy(outcome.capability.executable, eval_sets(plan.name, plan.arity, seed))
        record["capabilities"][plan.name] = entry
        print(f"  {plan.name:<9} {'ok' if outcome.learned else 'not learned':<11} {last['strategy']:<5} "
              f"{last['train_examples']:>2} ex  {entry.get('accuracy', '')}  {last['structure']}", flush=True)

    controls: dict = {"kernel": kernel_is_task_agnostic([c.name for c in registry])}
    if "mul" in registry:
        controls["rerouting_mul"] = rerouting_control(registry, "mul", seed)
    path = out_dir / f"loop_registry_{seed}.json"
    registry.save(path)
    reloaded = Registry.load(path)
    probe = random_tuples(3, 12, 40, random.Random(seed))
    controls["reload_identical"] = all(
        reloaded.run(c.name, [t[: c.plan.arity] for t in probe], 12)
        == registry.run(c.name, [t[: c.plan.arity] for t in probe], 12)
        for c in registry
    )
    record["controls"] = controls
    record["explain"] = {n: registry.explain(n) for n in ("max", "mul", "mul_add") if n in registry}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curriculum", type=Path, default=HERE / "curricula" / "loops.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--out", type=Path, default=Path("results/loop_learning.json"))
    args = parser.parse_args()

    plans = [LearningPlan.from_dict(p) for p in json.loads(args.curriculum.read_text())["plans"]]
    runs = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        runs.append(run_seed(plans, seed, args.out.parent))
        c = runs[-1]["controls"]
        rr = c.get("rerouting_mul", {})
        print(f"  controls: kernel clean {c['kernel']['clean']}, reload identical {c['reload_identical']}, "
              f"mul re-routings changing behaviour {rr.get('behaviour_changed')}/{rr.get('reroutings')}",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "loop_learning", "runs": runs}, indent=1))

    print("\nsummary (learned seeds | strategies | examples per seed)")
    for plan in plans:
        rows = [r["capabilities"][plan.name] for r in runs]
        strategies = sorted({r["strategy"] for r in rows if r["learned"]}) or ["-"]
        print(f"  {plan.name:<9} {sum(r['learned'] for r in rows)}/{len(rows)}  {','.join(strategies):<12} "
              f"{[r['examples'] if r['learned'] else 'x' for r in rows]}")
    print("\n" + runs[0]["explain"].get("mul", ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
