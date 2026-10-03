"""M1: does the shared-transition approach extend beyond addition?

Every operation gets the same budget: 32 of the 256 4-bit operand pairs,
supervised on final results only, with a human-provided schema. Each is then
verified against its local rule and evaluated up to 64 bits.

Multiplication is tested three ways:
  * mul_e2e:    gate (map) and adder (scan) cells trained jointly from 32 products
  * mul_e2e_r5: same, best of 5 initializations by training loss only
  * mul_reuse:  the AND and ADD units learned above, composed with no training

Usage: python experiments/primitives.py [--ops ...] [--seeds ...] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from vectorpro.benchmark import OperationSpec, WidthSuite, evaluate_modes, run_spec, summarize
from vectorpro.cells import CellSignature, MLPCell
from vectorpro.programs import ShiftAddMultiply
from vectorpro.schemas import MapSchema, ScanSchema
from vectorpro.tasks import (
    AND,
    FULL_ADDER,
    FULL_SUBTRACTOR,
    LESS_THAN,
    OR,
    XOR,
    Addition,
    Bitwise,
    LessThan,
    Multiplication,
    Subtraction,
)
from vectorpro.training import TrainConfig
from vectorpro.units import FunctionUnit, iter_units

MUL_SUITE = WidthSuite(n_random=1_000)


def scan_unit(name: str, signature: CellSignature, initial_state: tuple[int, ...]) -> FunctionUnit:
    return FunctionUnit(name, ScanSchema(2, initial_state), MLPCell(signature))


def map_unit(name: str) -> FunctionUnit:
    return FunctionUnit(name, MapSchema(2), MLPCell(CellSignature(2, 0, 1)))


def bitwise_spec(task: Bitwise) -> OperationSpec:
    return OperationSpec(task.name, task, lambda: map_unit(task.name), {task.name: task.local_rule})


def build_mul() -> ShiftAddMultiply:
    return ShiftAddMultiply(map_unit("gate"), scan_unit("adder", FULL_ADDER.signature, (0,)))


SPECS = {
    "add": OperationSpec("add", Addition(), lambda: scan_unit("add", FULL_ADDER.signature, (0,)),
                         {"add": FULL_ADDER}),
    "sub": OperationSpec("sub", Subtraction(), lambda: scan_unit("sub", FULL_SUBTRACTOR.signature, (0,)),
                         {"sub": FULL_SUBTRACTOR}),
    "lt": OperationSpec("lt", LessThan(), lambda: scan_unit("lt", LESS_THAN.signature, (0,)),
                        {"lt": LESS_THAN}),
    "and": bitwise_spec(AND),
    "or": bitwise_spec(OR),
    "xor": bitwise_spec(XOR),
    "mul_e2e": OperationSpec("mul_e2e", Multiplication(), build_mul,
                             {"gate": AND.local_rule, "adder": FULL_ADDER}, MUL_SUITE),
    "mul_e2e_r5": OperationSpec("mul_e2e_r5", Multiplication(), build_mul,
                                {"gate": AND.local_rule, "adder": FULL_ADDER}, MUL_SUITE,
                                restarts=5),
}


def run_mul_reuse(add_unit, and_unit, seed: int) -> dict:
    """Compose already-trained units into a multiplier; no further training.

    Evaluated on the same cases ``mul_e2e`` sees for this seed.
    """
    program = ShiftAddMultiply(and_unit, add_unit)
    rng = random.Random(seed)
    _, unseen = MUL_SUITE.split(2, rng)
    cases = MUL_SUITE.cases(2, unseen, rng)
    return {
        "operation": "mul_reuse",
        "seed": seed,
        "train": None,
        "verification": {u.name: u.verification.to_dict() for u in iter_units(program)},
        "evaluations": evaluate_modes(program, Multiplication(), cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", nargs="+", default=[*SPECS, "mul_reuse"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--out", type=Path, default=Path("results/primitives.json"))
    args = parser.parse_args()

    config = TrainConfig(steps=args.steps)
    needs = set(args.ops) | ({"add", "and"} if "mul_reuse" in args.ops else set())
    records: dict[str, list[dict]] = {}
    trained: dict[tuple[str, int], FunctionUnit] = {}
    for name in [n for n in SPECS if n in needs]:
        for seed in args.seeds:
            run = run_spec(SPECS[name], seed, config)
            trained[name, seed] = run.executable
            if name in args.ops:
                records.setdefault(name, []).append(run.record)
        if name in records:
            print(summarize(records[name]), flush=True)

    if "mul_reuse" in args.ops:
        records["mul_reuse"] = [
            run_mul_reuse(trained["add", s], trained["and", s], s) for s in args.seeds
        ]
        print(summarize(records["mul_reuse"]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "primitives", "operations": records}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
