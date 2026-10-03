# vectorpro

Vector programs executed by learned, verifiable state-transition cells.

The goal: people state intent, a language model turns it into a **learning
plan** (what to learn, from which targets, when it is done), and the program
itself is **learned, not coded**. Learned programs accumulate in a registry
that can be searched, explained in plain language, and asked to run.

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
  programs/        wiring-only compositions that stay in tensor space:
                   Fold, Window, ShiftAddMultiply, RestoringDivide, ExpressionProgram
  expr.py          expression AST shared by programs (execution) and tasks (reference)
  tasks/           Task (reference semantics) + LocalRule (verification oracle)
                   add, sub, lt, and/or/xor, mux, mul, divmod, expressions
  training.py      Trainer: end-to-end through any executable, final results only
  verification.py  exhaustive local-rule check
  evaluation.py    executor-agnostic harness
  benchmark.py     OperationSpec + run_spec: one train/verify/evaluate protocol
  catalog.py       standard learnable primitive units and their specs
  faults.py        single-entry fault injection for negative controls
  data.py          operand sets, splits, edge cases
  learning/        the learning loop (M2):
    plan.py        LearningPlan: plain-data plan (what an LLM would emit)
    examples.py    target sources, example streams; the learner's only view of targets
    search.py      composition search over registered capabilities
    learner.py     reuse first, else learn a unit over generic structures, round by round
    registry.py    capabilities stored as data; search, explain, run, save/load
experiments/       declare specs or curricula, write JSON to results/
  curricula/       learning plans as JSON
```

Boundaries:

- **Tasks never execute.** They give reference results, plus optional local rules used only for verification.
- **Schemas own structure, cells own the rule.** In M0–M1b schemas are given; from M2 the learner chooses among generic ones.
- **Programs only wire.** Shift, slice, concatenate, broadcast and constants are allowed; every bit of computation goes through a unit, so a program is exactly as correct as its verifiable units.
- **Integers appear only at the encode/decode boundary** of `BitExecutable.__call__`.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python experiments/add_width_generalization.py   # M0
python experiments/primitives.py                  # M1
python experiments/arithmetic.py                  # M1b
python experiments/learning_loop.py               # M2
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

## M1b result: the four arithmetic operations

Four units are learned, each from 32 4-bit examples with final results only,
and verified exhaustively: add, sub, and, and mux (bitwise select). Everything
else is wired from them with **no further training**
(`python experiments/arithmetic.py`, seeds 20261002–4).

| operation | built from | 4-bit, 8-bit exhaustive | 16/32/64-bit random + edge |
|---|---|---|---|
| `+` | add unit | 100% | 100% |
| `-` | sub unit | 100% | 100% |
| `*` | shift-add(and, add) | 100% | 100% |
| `/`, `%` | restoring division(sub, mux) | 100% | 100% |

These scores hold in all three execution modes. Division uses the sub unit's
borrow-out as its comparison and the mux unit for both restoring and emitting
quotient bits. Division by zero follows RISC-V: all-ones quotient, dividend as
remainder.

**Unseen programs.** 300 random expression trees over `+ - * / %` (1–7
operators, mean 4.3) are evaluated at 32-bit `uint32_t` semantics on 128
inputs each, half of them small values so that division by zero occurs. All
300 expressions and all 38,400 samples are exact in every mode.

**Negative control.** Each of the 44 single table entries across the four
units is flipped in turn. Every fault is flagged, and exactly by the
operations that use that unit: add 16, sub 16, mul 20 (and + add), divmod 24
(sub + mux). No fault goes unnoticed, so the 100% results are not an artifact
of a lenient harness.

Scope: the operand structure (bit order, shift-add, restoring division) is
human-provided, values are unsigned, and the e2e failure seen for
multiplication in M1 was not retried for division.

## M2 result: the learning loop

No per-task code and no per-task structure. A curriculum of 17 plans
(`experiments/curricula/arithmetic.json`, plain data) is fed to one learner in
order. For each plan, the learner draws examples round by round: 8, 16, 32, 64
at 4 bits, plus 32 validation examples at 8 bits. It first searches for a
composition of what the registry already holds. Otherwise it learns a new unit
by trying generic structures (map, scan from the low or high bit), simplest
first. It accepts a candidate once its table form reproduces every example,
then registers it. The learner never sees target functions or local rules,
only the examples it asked for (`python experiments/learning_loop.py`, seeds
20261002–4).

| plan | learned | how | examples used |
|---|---|---|---|
| and, or, xor, xnor, mux | 3/3 each | new unit (map) | 8 |
| add, sub | 3/3 each | new unit (low-to-high scan, carry) | 8–32 |
| lt, gt | 3/3 each | new unit (low-to-high scan, final state) | 8–16 |
| double, add3, sub_add, triple_sub, sum_xor | 3/3 each | **reuse**: composition found, no training | 8 |
| mul, avg, max | 0/3 | not learned within 64 examples | – |

- **Structure was chosen, not given.** The learner picked map for bitwise tasks
  and a low-to-high scan for add, sub, lt and gt.
- **Learned rules are exact.** An experimenter-only audit compared all 7 audited
  unit tables (and, or, xor, mux, add, sub, lt) with the known rules: 21/21
  exact over 3 seeds. Every learned capability is 100% on 16- and 32-bit probes.
- **Accuracy rises with data.** For example, `lt` on seed 20261002 had 88%
  validation from 8 examples, then 100% from 16. Every round is logged with
  its examples, structure, accuracies and probe.
- **Reuse grows with the registry.** Later tasks were composed from earlier
  ones, e.g. `triple_sub` = `((x0 add x0) add (x0 sub x1))`, with zero training.
- **No forgetting.** Learning never modifies registered capabilities, and all
  of them behave identically after the full curriculum.
- **Requests.** The saved registry (`results/registry.json`, 16 KB of data and
  no code) reloads and answers text search (`"sum"` → add, add3, sum_xor), lookup
  by unnamed examples (→ add3) and runs at 32 bits. `Capability.describe()`
  explains each capability in plain language, including its truth table.

**Where it stops.** mul, avg and max need structure the generic candidates do
not have: a loop over bits (mul), information flowing from higher bits (avg),
or a decision broadcast across all bits (max, which plateaus near 50%). These
are the next capabilities to add to the *machine*, not to any one task.

## Roadmap

| | milestone |
|---|---|
| M0 | core abstractions + addition reproduction ✅ |
| M1 | sub, compare, bitwise, multiply at an equal data budget ✅ |
| M1b | four arithmetic operations + random expressions from four verified units ✅ |
| M2 | learning loop: plans → reuse/learn → searchable, explainable registry ✅ |
| M2b | generic structures for loops, high-to-low flow and broadcast (mul, avg, max); LLM-written plans |
| M3 | structure discovery beyond a fixed candidate set; capability search by behaviour vectors |
| M4 | ingest binaries via emulator traces (Unicorn), held-out compilers |
| M5 | synthesizer, effect log for native calls, cost vs native/interpreter |
