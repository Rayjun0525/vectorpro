"""M1b: the four arithmetic operations, and expressions over them, from learned units.

1. Learn four primitive units, each from 32 4-bit examples with final results
   only, and verify each exhaustively: add, sub, and, mux.
2. Wire them into the four operations with no further training:
     +  add unit                   -  sub unit
     *  shift-add(and, add)        /, %  restoring division(sub, mux)
   and evaluate 4- and 8-bit exhaustively, 16/32/64-bit on random + edge pairs.
   Division by zero follows RISC-V (quotient all ones, remainder = dividend).
3. Evaluate random expression trees over + - * / % with W-bit unsigned
   semantics (programs never seen in training) at 32 bits.
4. Negative control: flip each single entry of each unit's table and check
   which operations the harness then reports as broken (16-bit cases).

Usage: python experiments/arithmetic.py [--seeds ...] [--expressions N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from vectorpro.benchmark import DEFAULT_MODES, evaluate_modes, run_spec
from vectorpro.catalog import UNIT_SPECS
from vectorpro.data import all_tuples, edge_pairs, random_tuples
from vectorpro.evaluation import EvalCase
from vectorpro.execution import BitExecutable
from vectorpro.expr import count_ops, random_expression, render
from vectorpro.faults import single_faults
from vectorpro.programs import ExpressionProgram, RestoringDivide, ShiftAddMultiply, Window
from vectorpro.tasks import Addition, DivMod, ExpressionTask, Multiplication, Subtraction, Task
from vectorpro.training import TrainConfig
from vectorpro.units import FunctionUnit

UNITS = ("add", "sub", "and", "mux")
N_RANDOM = 1_000
EXPR_WIDTH, EXPR_VARS, EXPR_DEPTH = 32, 4, 3
EXPR_OPS = ("+", "-", "*", "/", "%")
TABLE_MODE = next(m for m in DEFAULT_MODES if m.compile)


def build_operations(units: dict[str, BitExecutable]) -> dict[str, tuple[BitExecutable, Task]]:
    divide = RestoringDivide(units["sub"], units["mux"])
    return {
        "add": (units["add"], Addition()),
        "sub": (units["sub"], Subtraction()),
        "mul": (ShiftAddMultiply(units["and"], units["add"]), Multiplication()),
        "divmod": (divide, DivMod()),
    }


def operator_table(units: dict[str, BitExecutable]) -> dict[str, BitExecutable]:
    """``W -> W`` executables with modular semantics, keyed by operator symbol."""
    ops = {name: executable for name, (executable, _) in build_operations(units).items()}
    return {
        "+": Window(ops["add"]),
        "-": Window(ops["sub"]),
        "*": Window(ops["mul"]),
        "/": Window(ops["divmod"], word=0),
        "%": Window(ops["divmod"], word=1),
    }


def operation_cases(rng: random.Random) -> list[EvalCase]:
    cases = [EvalCase(f"{w}bit_exhaustive", w, all_tuples(2, w)) for w in (4, 8)]
    for width in (16, 32, 64):
        cases.append(
            EvalCase(f"{width}bit_random+edge", width, random_tuples(2, width, N_RANDOM, rng) + edge_pairs(width))
        )
    return cases


def evaluate_expressions(units: dict[str, BitExecutable], n_expr: int, seed: int) -> dict:
    rng = random.Random(f"expressions-{seed}")
    table = operator_table(units)
    per_mode = {m.name: {"fully_correct": 0, "samples": 0, "correct": 0, "failures": []} for m in DEFAULT_MODES}
    sizes = []
    for _ in range(n_expr):
        expr = random_expression(rng, EXPR_VARS, EXPR_DEPTH, EXPR_OPS)
        sizes.append(count_ops(expr))
        # Full-width values plus small ones, so division by zero and small quotients occur.
        inputs = random_tuples(EXPR_VARS, EXPR_WIDTH, 64, rng) + random_tuples(EXPR_VARS, 4, 64, rng)
        program = ExpressionProgram(expr, table, EXPR_VARS)
        task = ExpressionTask(expr, EXPR_VARS)
        results = evaluate_modes(program, task, [EvalCase(render(expr), EXPR_WIDTH, inputs)])
        for mode, (result,) in results.items():
            stats = per_mode[mode]
            stats["samples"] += result["total"]
            stats["correct"] += result["correct"]
            if result["correct"] == result["total"]:
                stats["fully_correct"] += 1
            elif len(stats["failures"]) < 5:
                stats["failures"].append(result["name"])
    return {
        "width": EXPR_WIDTH,
        "expressions": n_expr,
        "ops_per_expression": {"min": min(sizes), "max": max(sizes), "mean": sum(sizes) / len(sizes)},
        "modes": per_mode,
    }


def fault_control(units: dict[str, FunctionUnit], rng: random.Random) -> dict:
    """For each single-entry fault, which operations does evaluation flag as wrong?"""
    width = 16
    case = [EvalCase("16bit_random+edge", width, random_tuples(2, width, N_RANDOM, rng) + edge_pairs(width))]
    compiled = {name: unit.compiled() for name, unit in units.items()}
    out = {}
    for name, unit in units.items():
        for label, faulty in single_faults(unit):
            ops = build_operations({**compiled, name: faulty})
            out[f"{name} {label}"] = {
                op: evaluate_modes(executable, task, case, [TABLE_MODE])[TABLE_MODE.name][0]["accuracy"]
                for op, (executable, task) in ops.items()
            }
    return out


def run_seed(seed: int, config: TrainConfig, n_expr: int) -> dict:
    unit_runs = {name: run_spec(UNIT_SPECS[name], seed, config) for name in UNITS}
    units = {name: run.executable for name, run in unit_runs.items()}
    rng = random.Random(f"operations-{seed}")
    cases = operation_cases(rng)
    return {
        "seed": seed,
        "units": {name: run.record for name, run in unit_runs.items()},
        "operations": {
            name: evaluate_modes(executable, task, cases)
            for name, (executable, task) in build_operations(units).items()
        },
        "expressions": evaluate_expressions(units, n_expr, seed),
        "fault_control": fault_control(units, random.Random(f"faults-{seed}")),
    }


def print_summary(runs: list[dict]) -> None:
    for name in UNITS:
        exact = sum(r["units"][name]["verification"][name]["exact"] for r in runs)
        print(f"unit {name}: local rule exact on {exact}/{len(runs)} seeds")
    for op in runs[0]["operations"]:
        print(f"== {op} (composed, no training) ==")
        for mode, cases in runs[0]["operations"][op].items():
            parts = []
            for i, case in enumerate(cases):
                correct = sum(r["operations"][op][mode][i]["correct"] for r in runs)
                total = sum(r["operations"][op][mode][i]["total"] for r in runs)
                parts.append(f"{case['name'].split('_')[0]} {correct}/{total}")
            print(f"  [{mode}] " + ", ".join(parts))
    print(f"== random expressions ({EXPR_WIDTH}-bit, depth <= {EXPR_DEPTH}) ==")
    for mode in runs[0]["expressions"]["modes"]:
        stats = [r["expressions"]["modes"][mode] for r in runs]
        exprs = sum(r["expressions"]["expressions"] for r in runs)
        print(f"  [{mode}] expressions fully correct {sum(s['fully_correct'] for s in stats)}/{exprs}, "
              f"samples {sum(s['correct'] for s in stats)}/{sum(s['samples'] for s in stats)}")
    faults = runs[0]["fault_control"]
    print(f"== fault control (seed {runs[0]['seed']}, {len(faults)} single-entry faults, 16-bit) ==")
    for op in runs[0]["operations"]:
        caught = sum(acc[op] < 1 for acc in faults.values())
        print(f"  {op:<7} flagged by {caught}/{len(faults)} faults")
    silent = [f for f, acc in faults.items() if all(a == 1 for a in acc.values())]
    print(f"  faults no operation detects: {silent or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--expressions", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/arithmetic.json"))
    args = parser.parse_args()

    config = TrainConfig(steps=args.steps)
    runs = []
    for seed in args.seeds:
        runs.append(run_seed(seed, config, args.expressions))
        print(f"seed {seed} done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "arithmetic", "runs": runs}, indent=2))
    print_summary(runs)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
