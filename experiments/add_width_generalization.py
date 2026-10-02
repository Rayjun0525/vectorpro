"""M0: reproduce shared-bit-transition width generalization for addition.

Train an MLP cell inside an LSB-first scan on 32 of the 256 4-bit pairs, with
supervision on final sums only. Then:
  * verify the cell's binary local rule exhaustively against a full adder,
  * evaluate soft, quantized and table-compiled execution up to 64 bits,
  * run a 4-step fold (five 32-bit operands, sum mod 2**32).

Usage: python experiments/add_width_generalization.py [--seeds ...] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import torch

from vectorpro.cells import MLPCell
from vectorpro.data import all_tuples, edge_pairs, random_tuples, split
from vectorpro.evaluation import EvalCase, evaluate
from vectorpro.execution import BitExecutable
from vectorpro.programs import Fold
from vectorpro.quantize import HardThreshold, Identity
from vectorpro.schemas import ScanSchema
from vectorpro.tasks import FULL_ADDER, Addition, ModularSum
from vectorpro.training import TrainConfig, Trainer
from vectorpro.units import FunctionUnit
from vectorpro.verification import verify_local_rule

TRAIN_WIDTH = 4
N_TRAIN = 32
N_RANDOM = 10_000
FOLD_WIDTH, FOLD_TERMS, N_FOLD = 32, 5, 1_000


def build_cases(test_pairs, rng: random.Random) -> list[EvalCase]:
    cases = [
        EvalCase("4bit_unseen", TRAIN_WIDTH, test_pairs),
        EvalCase("8bit_exhaustive", 8, all_tuples(2, 8)),
    ]
    for width in (16, 32, 64):
        operands = random_tuples(2, width, N_RANDOM, rng) + edge_pairs(width)
        cases.append(EvalCase(f"{width}bit_random+edge", width, operands))
    return cases


def run_seed(seed: int, config: TrainConfig) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    task = Addition()

    train_pairs, test_pairs = split(all_tuples(2, TRAIN_WIDTH), N_TRAIN, rng)
    unit = FunctionUnit("add", ScanSchema(arity=2, initial_state=(0,)), MLPCell(FULL_ADDER.signature))
    report = Trainer(config).fit(unit, task, train_pairs, TRAIN_WIDTH)
    unit.verification = verify_local_rule(unit.cell, FULL_ADDER)
    compiled = unit.compiled()

    executors: dict[str, BitExecutable] = {"neural": unit, "table": compiled}
    quantizers = {"continuous": Identity(), "quantized": HardThreshold()}
    cases = build_cases(test_pairs, rng)
    fold_case = EvalCase(
        f"fold{FOLD_TERMS}x{FOLD_WIDTH}bit", FOLD_WIDTH, random_tuples(FOLD_TERMS, FOLD_WIDTH, N_FOLD, rng)
    )
    fold_task = ModularSum(FOLD_TERMS)

    evaluations = {}
    for exec_name, executable in executors.items():
        for q_name, quantizer in quantizers.items():
            if exec_name == "table" and q_name == "continuous":
                continue  # a table cell is binary by construction
            run = partial(executable, quantizer=quantizer)
            fold = partial(Fold(executable, FOLD_TERMS), quantizer=quantizer)
            results = evaluate(run, task.reference, cases)
            results += evaluate(fold, fold_task.reference, [fold_case])
            evaluations[f"{exec_name}/{q_name}"] = [r.to_dict() for r in results]

    return {
        "seed": seed,
        "train": {"width": TRAIN_WIDTH, "pairs": N_TRAIN, **vars(report)},
        "verification": unit.verification.to_dict(),
        "evaluations": evaluations,
    }


def print_summary(runs: list[dict]) -> None:
    print("\nlocal rule exact:", {r["seed"]: r["verification"]["exact"] for r in runs})
    for mode in runs[0]["evaluations"]:
        print(f"\n[{mode}] (summed over seeds)")
        for i, case in enumerate(runs[0]["evaluations"][mode]):
            correct = sum(r["evaluations"][mode][i]["correct"] for r in runs)
            total = sum(r["evaluations"][mode][i]["total"] for r in runs)
            print(f"  {case['name']:<22} {correct:>7}/{total:<7}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--out", type=Path, default=Path("results/add_width_generalization.json"))
    args = parser.parse_args()

    config = TrainConfig(steps=args.steps)
    runs = [run_seed(seed, config) for seed in args.seeds]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "add_width_generalization", "runs": runs}, indent=2))
    print_summary(runs)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
