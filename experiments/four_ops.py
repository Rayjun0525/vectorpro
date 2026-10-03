"""M2d: the four arithmetic operations (and remainder) learned from examples only.

One learner works through experiments/curricula/four_ops.json. New machine
features since M2c, all task-agnostic:
  * register headroom: programs may compute on registers wider than W bits
    (e.g. avg = shr(x0 add x1) with one extra bit);
  * companion accumulators: a loop may run alongside a loop already learned,
    with only its own update searched. div is found beside the remainder
    loop learned for mod.

Then the learned +, -, *, /, % are used as operators of random expression
trees, compiled into vector programs and run at 32 bits: unseen programs
built only from learned parts.

Controls: accuracy at 8/16/32 bits; every single re-routing of the learned
div program (first seed); kernel names no capability; registry reloads.

Usage: python experiments/four_ops.py [--seeds ...] [--expressions N]
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
from vectorpro.expr import BinOp, count_ops, random_expression, render
from vectorpro.learning import Capability, ExampleSet, Learner, LearningPlan, OutputWidth, Registry
from vectorpro.learning.registry import program_provenance
from vectorpro.machine import compile_expression
from vectorpro.tasks import ExpressionTask

HERE = Path(__file__).parent
SYMBOLS = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod"}
EXPR_WIDTH, EXPR_VARS, EXPR_DEPTH = 32, 4, 3


def rename(expr, mapping):
    if isinstance(expr, BinOp):
        return BinOp(mapping[expr.op], rename(expr.left, mapping), rename(expr.right, mapping))
    return expr


def expression_check(registry: Registry, n_expr: int, seed: int) -> dict:
    """Random + - * / % trees, compiled to vector programs over the learned capabilities."""
    rng = random.Random(f"expressions-{seed}")
    exprs_ok = samples = correct = 0
    failures = []
    sizes = []
    for k in range(n_expr):
        expr = random_expression(rng, EXPR_VARS, EXPR_DEPTH, sorted(SYMBOLS))
        sizes.append(count_ops(expr))
        program = compile_expression(rename(expr, SYMBOLS), EXPR_VARS, registry.key_of)
        plan = LearningPlan(f"expr{k}", render(expr), EXPR_VARS, OutputWidth.SAME)
        executable = registry.build(plan, program_provenance(program, "expression check"))
        inputs = random_tuples(EXPR_VARS, EXPR_WIDTH, 48, rng) + random_tuples(EXPR_VARS, 3, 16, rng)
        task = ExpressionTask(expr, EXPR_VARS)
        got = executable(inputs, EXPR_WIDTH)
        ok = sum(g == task.reference(t, EXPR_WIDTH) for g, t in zip(got, inputs))
        samples += len(inputs)
        correct += ok
        if ok == len(inputs):
            exprs_ok += 1
        elif len(failures) < 5:
            failures.append(render(expr))
    return {"expressions": n_expr, "fully_correct": exprs_ok, "samples": samples, "correct": correct,
            "ops_per_expression": sum(sizes) / len(sizes), "failures": failures}


def run_seed(plans: list[LearningPlan], seed: int, n_expr: int, out_dir: Path, controls: bool) -> dict:
    torch.manual_seed(seed)
    registry = Registry(seed=seed)
    learner = Learner(registry)
    record: dict = {"seed": seed, "capabilities": {}}
    for plan in plans:
        outcome = learner.learn(plan, TARGETS[plan.name], random.Random(f"{plan.name}-{seed}"))
        last = outcome.history[-1]
        entry = {"learned": outcome.learned, "strategy": last["strategy"], "structure": last["structure"],
                 "examples": last["train_examples"], "rounds": len(outcome.history)}
        if outcome.learned:
            entry["accuracy"] = accuracy(outcome.capability.executable, eval_sets(plan.name, plan.arity, seed))
        record["capabilities"][plan.name] = entry
        print(f"  {plan.name:<7} {'ok' if outcome.learned else 'not learned':<11} {last['strategy']:<5} "
              f"{last['train_examples']:>2} ex  {entry.get('accuracy', '')}  {last['structure']}", flush=True)

    if all(registry.__contains__(n) for n in SYMBOLS.values()):
        record["expressions"] = expression_check(registry, n_expr, seed)
        e = record["expressions"]
        print(f"  expressions: {e['fully_correct']}/{e['expressions']} fully correct, "
              f"{e['correct']}/{e['samples']} samples", flush=True)
    checks: dict = {"kernel": kernel_is_task_agnostic([c.name for c in registry])}
    if controls and "div" in registry:
        checks["rerouting_div"] = rerouting_control(registry, "div", seed)
        rr = checks["rerouting_div"]
        print(f"  div re-routings changing behaviour: {rr['behaviour_changed']}/{rr['reroutings']}", flush=True)
    path = out_dir / f"four_ops_registry_{seed}.json"
    registry.save(path)
    reloaded = Registry.load(path)
    probe = random_tuples(2, 12, 40, random.Random(seed))
    checks["reload_identical"] = all(
        reloaded.run(c.name, [t[: c.plan.arity] for t in probe], 12)
        == registry.run(c.name, [t[: c.plan.arity] for t in probe], 12)
        for c in registry
    )
    record["controls"] = checks
    record["explain"] = {n: registry.explain(n) for n in ("div", "avg") if n in registry}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curriculum", type=Path, default=HERE / "curricula" / "four_ops.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--expressions", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("results/four_ops.json"))
    args = parser.parse_args()

    plans = [LearningPlan.from_dict(p) for p in json.loads(args.curriculum.read_text())["plans"]]
    runs = []
    for i, seed in enumerate(args.seeds):
        print(f"seed {seed}", flush=True)
        runs.append(run_seed(plans, seed, args.expressions, args.out.parent, controls=(i == 0)))
        print(f"  controls: kernel clean {runs[-1]['controls']['kernel']['clean']}, "
              f"reload identical {runs[-1]['controls']['reload_identical']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "four_ops", "runs": runs}, indent=1))

    print("\nsummary (learned seeds | strategies | examples per seed)")
    for plan in plans:
        rows = [r["capabilities"][plan.name] for r in runs]
        strategies = sorted({r["strategy"] for r in rows if r["learned"]}) or ["-"]
        print(f"  {plan.name:<7} {sum(r['learned'] for r in rows)}/{len(rows)}  {','.join(strategies):<12} "
              f"{[r['examples'] if r['learned'] else 'x' for r in rows]}")
    print("\n" + runs[0]["explain"].get("div", ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
