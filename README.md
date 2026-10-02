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
  execution.py     BitExecutable: tensor-in/tensor-out executable tree
                   (execute, compiled, children, parameters)
  units.py         FunctionUnit = schema + cell + verification record (leaf)
  programs.py      Fold, ShiftAddMultiply: compositions that stay in tensor space
  tasks/           Task (reference semantics) + LocalRule (verification oracle)
                   add, sub, lt, and/or/xor, mul, modular sum
  training.py      Trainer: end-to-end through any executable, final results only
  verification.py  exhaustive local-rule check
  evaluation.py    executor-agnostic harness
  benchmark.py     OperationSpec + run_spec: one train/verify/evaluate protocol
  data.py          operand sets, splits, edge cases
experiments/       declare specs, write JSON to results/
```

Boundaries:

- **Tasks never execute.** They give reference results, plus optional local rules used only for verification.
- **Schemas own structure, cells own the rule.** Schemas are currently human-provided. Discovering them is milestone M3.
- **Integers appear only at the encode/decode boundary** of `BitExecutable.__call__`.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python experiments/add_width_generalization.py   # M0
python experiments/primitives.py                  # M1
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

## M1 result: beyond addition

Same protocol and budget for every operation: 32 of the 256 4-bit pairs,
final results only, a human-provided schema, and seeds 20261002–4. Each
operation is evaluated on 4-bit unseen pairs, 8-bit exhaustive inputs and
16/32/64-bit random + edge inputs in all three execution modes
(`python experiments/primitives.py`).

| operation | schema | local rule exact | all cases, all modes |
|---|---|---|---|
| add | LSB scan, carry | 3/3 | 100% |
| sub (borrow-out in top bit) | LSB scan, borrow | 3/3 | 100% |
| lt | LSB scan, 1-bit state, final state only | 3/3 | 100% |
| and / or / xor | map | 3/3 each | 100% |
| mul_reuse: learned AND + ADD composed, no training | shift-add program | 3/3 | 100% |
| mul_e2e: gate + adder trained jointly from 32 products | shift-add program | gate 2/3, adder 1/3 | 1/3 seeds |
| mul_e2e_r5: best of 5 inits by training loss | shift-add program | gate 2/3, adder 1/3 | 1/3 seeds |

What this shows:

- Every single-cell operation learned its exact local rule from 32 results and
  extended to 64 bits. For `lt` only the final bit was supervised.
- **Composition of verified units works; joint learning through deep composition
  did not.** The multiplier assembled from separately learned AND and ADD units is
  exact. Trained jointly from products, only 3 of 15 initializations reached low
  loss, all on one seed. One seed (20261004) found a pseudo-solution: 90.6% train
  accuracy with an adder that depends on non-binary intermediate values.
  Continuous execution partly works there, while quantized and table execution
  score 0%. Exhaustive verification flags it.
- Implication for the design: learn and verify small units, then compose them.
  Prefer that to end-to-end training of large programs.

## Roadmap

| | milestone |
|---|---|
| M0 | core abstractions + addition reproduction ✅ |
| M1 | sub, compare, bitwise, multiply at an equal data budget ✅ |
| M2 | function registry, id vectors, program tensors and an execution kernel |
| M3 | schema search: train small, select by generalization to larger widths |
| M4 | ingest binaries via emulator traces (Unicorn), held-out compilers |
| M5 | synthesizer, effect log for native calls, cost vs native/interpreter |
