# vectorpro

Vector programs executed by learned, verifiable state-transition cells.

A function is expressed as a **schema** (the iteration structure) applied to a
small learned **cell** (the shared local rule). Because a cell's binary domain
is finite, it can be verified exhaustively. An exactly verified cell inside a
correct schema is correct at every width, and it can be compiled to a lookup
table.

## Layout

```
src/vectorpro/
  bits.py          BitCodec: int <-> little-endian bit tensors
  quantize.py      Quantizer: Identity (soft) / HardThreshold (straight-through)
  cells/           Cell ABC + CellSignature; MLPCell (learned), TableCell (exact)
  schemas/         Schema ABC; ScanSchema (LSB/MSB, carried state), MapSchema
  execution.py     BitExecutable: tensor-in/tensor-out executable interface
  units.py         FunctionUnit = schema + cell + verification record
  programs.py      Fold: compositions that stay in tensor space
  tasks/           Task (reference semantics) + LocalRule (verification oracle)
  training.py      Trainer: end-to-end on final results only
  verification.py  exhaustive local-rule check
  evaluation.py    executor-agnostic harness
  data.py          operand sets, splits, edge cases
experiments/       runnable experiments, write JSON to results/
```

Boundaries:

- **Tasks never execute.** They give reference results, plus optional local rules used only for verification.
- **Schemas own structure, cells own the rule.** Schemas are currently human-provided. Discovering them is milestone M3.
- **Integers appear only at the encode/decode boundary** of `BitExecutable.__call__`.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python experiments/add_width_generalization.py
```

## M0 result: addition width generalization

An MLP cell inside an LSB-first scan is trained on 32 of the 256 4-bit pairs,
with supervision on final sums only. Totals are summed over seeds
20261002, 20261003 and 20261004. Every execution mode (soft, quantized,
table-compiled) scored:

| case | correct |
|---|---|
| local rule vs full adder (8 entries) | exact, 3/3 seeds |
| 4-bit unseen pairs | 672/672 |
| 8-bit exhaustive | 196,608/196,608 |
| 16-bit random + edge | 30,987/30,987 |
| 32-bit random + edge | 31,995/31,995 |
| 64-bit random + edge | 34,011/34,011 |
| fold, five 32-bit operands (mod 2^32) | 3,000/3,000 |

The scan structure is supplied by a human, so this does not show structure
discovery.

## Roadmap

| | milestone |
|---|---|
| M0 | core abstractions + addition reproduction ✅ |
| M1 | sub, compare, bitwise, multiply (nested scan) at an equal data budget |
| M2 | function registry, id vectors, program tensors and an execution kernel |
| M3 | schema search: train small, select by generalization to larger widths |
| M4 | ingest binaries via emulator traces (Unicorn), held-out compilers |
| M5 | synthesizer, effect log for native calls, cost vs native/interpreter |
