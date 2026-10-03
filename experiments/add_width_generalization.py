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
from pathlib import Path

from vectorpro.benchmark import Derived, OperationSpec, run_spec, summarize
from vectorpro.cells import MLPCell
from vectorpro.data import random_tuples
from vectorpro.evaluation import EvalCase
from vectorpro.programs import Fold
from vectorpro.schemas import ScanSchema
from vectorpro.tasks import FULL_ADDER, Addition, ModularSum
from vectorpro.training import TrainConfig
from vectorpro.units import FunctionUnit

FOLD_WIDTH, FOLD_TERMS, N_FOLD = 32, 5, 1_000

ADD_SPEC = OperationSpec(
    name="add",
    task=Addition(),
    build=lambda: FunctionUnit("add", ScanSchema(2, (0,)), MLPCell(FULL_ADDER.signature)),
    rules={"add": FULL_ADDER},
    derived=[
        Derived(
            build=lambda unit: Fold(unit, FOLD_TERMS),
            task=ModularSum(FOLD_TERMS),
            cases=lambda rng: [
                EvalCase(f"fold{FOLD_TERMS}x{FOLD_WIDTH}bit", FOLD_WIDTH,
                         random_tuples(FOLD_TERMS, FOLD_WIDTH, N_FOLD, rng))
            ],
        )
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--out", type=Path, default=Path("results/add_width_generalization.json"))
    args = parser.parse_args()

    config = TrainConfig(steps=args.steps)
    runs = [run_spec(ADD_SPEC, seed, config).record for seed in args.seeds]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "add_width_generalization", "runs": runs}, indent=2))
    print(summarize(runs))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
