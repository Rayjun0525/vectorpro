import random

import torch

from vectorpro.cells import MLPCell, TableCell
from vectorpro.data import all_tuples, random_tuples, split
from vectorpro.schemas import ScanSchema
from vectorpro.tasks import FULL_ADDER, Addition
from vectorpro.training import TrainConfig, Trainer
from vectorpro.units import FunctionUnit
from vectorpro.verification import verify_local_rule


def test_verification_reports_mismatch():
    table = TableCell.from_rule(FULL_ADDER)
    table.table[3, 0] = 1 - table.table[3, 0]
    record = verify_local_rule(table, FULL_ADDER)
    assert not record.exact
    assert len(record.mismatches) == 1
    assert record.mismatches[0].inputs == (1, 1) and record.mismatches[0].state == (0,)


def test_learns_full_adder_from_32_sums_and_scales():
    rng = random.Random(20261002)
    torch.manual_seed(20261002)
    train, _ = split(all_tuples(2, 4), 32, rng)
    unit = FunctionUnit("add", ScanSchema(2, (0,)), MLPCell(FULL_ADDER.signature))

    report = Trainer(TrainConfig(steps=2000)).fit(unit, Addition(), train, 4)

    assert report.train_exact_match == 1.0
    assert verify_local_rule(unit.cell, FULL_ADDER).exact
    ops = random_tuples(2, 64, 200, rng)
    assert unit.compiled()(ops, 64) == [a + b for a, b in ops]
