# vectorpro

실제 Gemma로 반복 작업의 비용을 비교했다. 정확한 구조화 입력의 텐서 직접 호출은
LLM 사용 0회였지만, 자연어 인자 준비는 오류가 많았고 Python 직접 함수가 더 빨랐다.
측정값과 조건은 [AGENT_COST.md](docs/AGENT_COST.md)를 참고한다.

첫 완성 목표는 **텐서 프로그램으로 리눅스의 파일·디렉터리·프로세스·입출력·네트워크를
사용하는 것**이다. 현재 파일시스템 기반을 구현했고, 한 텐서에 학습한 7개 기능을
리눅스의 미사용 루트 4개에서 재학습 없이 검증했다. 기준과 남은 범위는
[LINUX_TARGET.md](docs/LINUX_TARGET.md)를 참고한다.

계약 초안의 격리 학습·검증·등록과 이름 기반 인자 호출은
[CONTRACT_LEARNING.md](docs/CONTRACT_LEARNING.md)를 참고한다.

현재 로컬 LLM 기준은 **Gemma 3 1B Instruct Q8_0**이다. Laya 적합성 시험 후
사용자 요청으로 Laya 가중치를 제거하고 Gemma를 복원했다. 기존 컨테이너 하나를
재사용한다. 명시된 레지스터 폭이 하나인 요청은 그 폭만 도구 선택지로 제공한다.
최적화 범위·검증·남은 과제는 [GEMMA_BASELINE.md](docs/GEMMA_BASELINE.md),
Laya 과거 시험은 [LAYA.md](docs/LAYA.md)에 있다.

배운 기능은 공통 계약 ID와 이름 있는 typed 입력으로 LLM 없이 직접 호출할 수 있다.
계약은 같은 프로그램 파일에 저장하며 카탈로그 어댑터도 이 계약을 사용한다.
API/CLI/도구 schema/현재 한계는 [FUNCTION_CONTRACTS.md](docs/FUNCTION_CONTRACTS.md)를 참고한다.

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
  machine/         the vector machine (M2b):
    program.py     VectorProgram: a program's whole logic as tensors
    kernel.py      task-agnostic fetch-execute kernel; ProgramExecutable
    assembler.py   assemble / compile_expression (writing) and disassemble (reading)
    skeletons.py   generic loop skeletons (BitFold, with companions) compiled to vector programs
  programs/        Python-wired compositions used by M0–M1b experiments (legacy):
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
    examples.py    target sources, example streams (with boundary inputs); the learner's only view
    search.py      size-ordered composition search (any arity, constants, behaviour dedup)
    loops.py       loop search: bit-fold bodies, companion loops from the library; primitives by behaviour
    learner.py     per round: reuse, else learn a unit, else learn a loop
    registry.py    capabilities with address vectors; resolves calls, search, explain, save/load
experiments/       declare specs or curricula, write JSON to results/
  curricula/       learning plans as JSON
  authored.py      hand-written vector programs (mul, div, mod): assembly-level demos
  targets.py       target values standing in for human labels
```

Boundaries:

- **All program logic is in vectors.** A program is a `VectorProgram`: address
  vectors for the capabilities it calls, read and write matrices for arguments
  and results, and control tensors for branches and loops. Capabilities call
  each other by address vector, never by name. The kernel that runs programs
  holds no task knowledge; it is the CPU, not the program.
- **Tasks never execute.** They give reference results, plus optional local rules used only for verification.
- **Schemas own structure, cells own the rule.** In M0–M1b schemas are given; from M2 the learner chooses among generic ones.
- **Legacy Python programs only wire.** In M0–M1b, `programs/` classes did wiring only; from M2b this role moves into vector programs.
- **Integers appear only at the encode/decode boundary** of `BitExecutable.__call__`.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python experiments/add_width_generalization.py   # M0
python experiments/primitives.py                  # M1
python experiments/arithmetic.py                  # M1b
python experiments/learning_loop.py               # M2
python experiments/vector_programs.py             # M2b
python experiments/loop_learning.py               # M2c
python experiments/four_ops.py                    # M2d
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
- **Requests.** The saved registry (`results/registry.json`, data only and no
  code; compositions are stored as vector programs since M2b) reloads and answers text search (`"sum"` → add, add3, sum_xor), lookup
  by unnamed examples (→ add3) and runs at 32 bits. `Capability.describe()`
  explains each capability in plain language, including its truth table.

**Where it stops.** mul, avg and max need structure the generic candidates do
not have: a loop over bits (mul), information flowing from higher bits (avg),
or a decision broadcast across all bits (max, which plateaus near 50%). These
are the next capabilities to add to the *machine*, not to any one task.

## M2b result: all program logic in vectors

A program is now a `VectorProgram`, a set of tensors. Each step holds the
**address vector** of the capability it calls, **read/write matrices** that
route arguments and results between registers, and **control tensors** for
conditional jumps and halting. A single kernel runs every program. It resolves
address vectors to capabilities by similarity, and its sources contain no
capability names. Capabilities call one another only by address, so the network
of vectors plays the role of functions and classes. Compositions the learner
finds are stored as such programs (`python experiments/vector_programs.py`,
seeds 20261002–4).

| capability | how it exists | 8/16/32-bit accuracy |
|---|---|---|
| and, or, xor, add, sub, lt, shr | learned units, 8–32 examples | 100% |
| shl | found by reuse as `x add x` (8 examples) | 100% |
| mul, div, mod | **hand-authored** loop/branch vector programs calling the learned units | 100% |
| square, mul_add, sum_mod, diff_div | found by reuse from 8 examples: vector programs calling mul/div/mod | 100% |

Controls, on every seed:

- **Dispatch is by vector.** Swapping the address vectors of add and sub leaves
  the mul program's tensors untouched but drops it to 3–8%.
- **Every routing entry carries logic.** Each of the 64 single re-routings of
  the mul program (one read, write or jump moved) changes its behaviour.
- **The kernel is task-agnostic.** Its sources name no capability.
- **Programs are durable data.** Registries reload from JSON with identical results,
  and `explain()` disassembles any program from its tensors, e.g. `mul_add`:
  `@0: r3 = mul(r0, r1)`, `@1: r4 = add(r2, r3)`.

Not yet learned: the loop programs themselves. mul, div and mod were written
by hand (`experiments/authored.py`, assembly-level). They show that the format
and kernel hold loops and branches. Finding such programs from examples is the
next step.

## M2c result: loop programs learned from examples

Nothing is authored. One learner works through 20 plans
(`experiments/curricula/loops.json`), each given only as examples. Each round
it tries, cheapest first:

1. a **composition** of registered capabilities. The search is size-ordered,
   handles any arity plus the constants `zero`, `one` and `ones`, and drops
   candidates that behave identically on the examples. Commutativity is
   detected by running each operator.
2. a **new unit**, as in M2.
3. a **loop**: the body of a generic *bit fold*, `acc = 0; for each bit of
   x[k] (high→low or low→high): acc = body(x…, acc, bit, mask)`. The machine
   provides the fold once. The bit-extraction capabilities it compiles to
   (and, lt, sub, shifts) are found in the registry **by behaviour**, not by
   name. The body is searched over units and straight-line programs, batched
   so that shared sub-expressions run once.

Results (`python experiments/loop_learning.py`, seeds 20261002–4). Every
learned capability is 100% at 8, 16 and 32 bits, and the same structure was
found on every seed:

| plan | how | what was found | examples |
|---|---|---|---|
| and, or, xor, sub, lt, shr | new unit | map / scans, as in M2 | 8–32 |
| add | new unit | low-to-high scan | 8–16 |
| shl | composition | `x add x` | 8 |
| mux | composition | `x0 xor (x2 and (x0 xor x1))` | 8 |
| neg, nand | composition | `zero sub x`, `mux(ones, zero, x0 and x1)` | 8 |
| max | composition | `mux(x0, x1, ones add (x1 lt x0))` | 8 |
| min | composition | `x0 add (x1 sub max(x0, x1))` | 8 |
| **mul** | **loop** | `for each bit of x1, high→low: acc = acc + (acc + (x0 and mask))` | 8 |
| **popcount** | **loop** | `for each bit, high→low: acc = acc + bit` | 8 |
| **reverse** | **loop** | `for each bit, low→high: acc = acc + (acc + bit)` | 8 |
| square, mul_add | composition calling the learned loop | `x0 mul x0`, `x2 add (x0 mul x1)` | 8 |
| avg, div | not learned | – | – |

- **mul was learned from 8 examples, as a loop.** The learner found Horner's
  method (`acc = 2·acc + (x0 if bit else 0)`), not the shift-and-add version
  authored in M2b. The result is an ordinary vector program; `explain("mul")`
  disassembles its 7 steps, including the backward jump.
- **Every routing entry of the learned mul carries logic.** All 236 single
  re-routings change its behaviour on every seed. The kernel names no
  capability, and registries reload identically.
- **Where it stops.** `avg` needs the carry out of the top bit inside a W-bit
  result. `div` needs two accumulators (quotient and remainder). Neither fits
  a single-accumulator fold or a size-3 composition. Loops inside loops are not
  searched; loop programs are composed only at the top level.

Results sections record the code at their milestone. Rerunning an earlier
experiment with later code can do better: with M2c's search, M2's `max` and
`mul` become learnable.

## M2d result: all four arithmetic operations learned from examples

`div` was the last operation still needing a hand-written program (M2b). It
is now learned from examples, together with everything it builds on
(`python experiments/four_ops.py`, `experiments/curricula/four_ops.json`,
seeds 20261002–4). Three machine-level additions, none task-specific:

- **Register headroom.** A program may compute on registers wider than `W`
  bits and return the low `W` bits, like a carry bit. Compositions and folds
  are tried at 0 and 1 extra bits.
- **Companion loops.** A loop can run alongside a loop the registry already
  learned, with only its own update searched. The update has the form
  `acc ← g(acc, h)` where `h` does not read `acc`. So `h` is enumerated with
  exact deduplication on the companion's real trajectory, and every
  `(g, h)` pair is tried in one batch.
- **Boundary examples.** Example draws include equal operands and
  0, 1, top-bit and all-ones values. Without them, `ge` was accepted as
  `x1 lt x0` (wrong only when `x0 = x1`) on one seed, because no random
  example had equal operands.

The 16 plans are all learned on every seed, and every capability is 100% at
8, 16 and 32 bits:

| plan | how | what was found | examples |
|---|---|---|---|
| and, or, xor, add, sub, lt, shr | new unit | map / scans | 8–16 |
| shl, dadd (`2·x0 + x1`) | composition | `x add x`, `x0 add (x0 add x1)` | 8 |
| ge, gemask, csub | composition | `(x0 lt x1) lt one`, `ones add (x0 lt x1)`, `x0 sub (x1 and gemask(x0, x1))` | 8 |
| **avg** | composition, **1 extra bit** | `shr(x0 add x1)` | 8 |
| **mul** | loop | `acc ← acc dadd (x0 and mask)` over x1, high→low | 8 |
| **mod** | loop | `acc ← (acc dadd bit) csub x1` over x0, high→low | 8 |
| **div** | loop **with the mod loop as companion** | `acc ← acc dadd ((c1 dadd bit) ge x1)`, `c1 ← (c1 dadd bit) csub x1` | 8 |

That is restoring long division, found from 8 examples. Division by zero
gives all ones and the remainder gives the dividend, matching RISC-V,
without special handling.

**Unseen programs.** 60 random expression trees per seed over the learned
`+ - * / %` are compiled into vector programs and run at 32 bits (48 random
and 16 small inputs each, so division by zero occurs). Results: **60/60 expressions and 3,840/3,840 samples exact on every seed**.

**Controls.** 255 of the 256 single re-routings of the learned `div` program
change its behaviour. The remaining one moves a read of the constant zero to an
unused register that also holds zero. Compiled folds now skip unused
bit-extraction steps and compute shared sub-expressions once, which removed
the dead code an earlier run exposed (348/376). On every seed, the kernel
names no capability and the registry reloads identically.

**Honest scope.** The helper plans (`dadd`, `ge`, `gemask`, `csub`) are
listed in the curriculum. Nothing in them names a structure, but choosing
these stepping stones is planning knowledge that a person or a language model
supplies. Loops inside loops are still not searched.

## Unified runtime and host execution foundation

`VectorRuntime` accepts a structured request directly. Known capability names
execute immediately without consulting a teacher. Unknown names enter learning
when a `LearningPlan` and example source are supplied, then execute on acceptance.
Without that evidence the runtime returns `needs_learning_examples`; failed
learning returns `learning_failed` without registering or executing a candidate.

```python
from pathlib import Path
from vectorpro.runtime import VectorRuntime
from vectorpro.learning import LearningPlan, OutputWidth

runtime = VectorRuntime(seed=0)
result = runtime.request(
    "add", [(7, 3)], 8,
    plan=LearningPlan("add", "sum with carry", 2, OutputWidth.PLUS_ONE),
    source=lambda operands, width: sum(operands),
)
# If learning succeeds, result.status == "learned_and_executed".
# Subsequent requests use the stored capability without a source.
runtime.save(Path("program.json"))
restored = VectorRuntime.load(Path("program.json"))
```

The single data file stores capability address vectors, learned tables, vector
programs and address/learning-sampler RNG states. It contains no Python source,
file handles or absolute host paths. Python, PyTorch and vectorpro are still
required to run it; standalone OS installers and LLM request interpretation
are not implemented. Old registries and programs remain loadable.

`HostContext(root)` adds transient byte-buffer handles and a relative-path file
backend. `runtime.provide_host_operations()` provides `buffer.new`,
`buffer.length`, `buffer.get`, `buffer.set`, `file.read`, and `file.write` as
execution primitives. These primitives are supplied, not learned. The root is
explicitly provided by the application; handles and buffers are recreated on
loading, rather than persisted in a program file.

Vector instructions now have a `calls` tensor independently of their write
destination: an operation can execute with its result discarded. Branch-only
steps have no call. This allows file writes and buffer mutation to actually
happen. Old programs infer call flags from their write destinations. Effectful
programs use one execution lane, produce an ordered host event log, and are
excluded from arithmetic composition search and example lookup so that learning
does not mutate files while trying candidates. OS errors and buffer bounds
errors propagate explicitly; completed effects are not rolled back on failure.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/runtime_requests.py
```

The demo learns addition, handles a known request, then runs an assembled tensor
workflow that reads two file bytes, adds them, updates a buffer and writes it
back. It saves one program file and repeats both computation and I/O after a
reload. The workflow itself is supplied: learning stateful I/O procedures and
ingesting binary execution traces remain future work. The native backend uses
Python's cross-platform file APIs but has only been tested in the existing
Linux container.

## Data-only requests and command-line execution

The runtime also learns from finite JSON input/output examples via
`ExampleLesson`, without a Python target callback. Lessons contain separate
training and validation sets and a shape/budget plan; they contain no program.
Inputs, output ranges, duplicate examples, validation counts, and train/validation
overlap are checked before learning. Candidates must match every supplied
training example as well as the validation set. Exhausted data produces a
learning failure rather than invented targets. Acceptance remains agreement
with supplied examples, not proof of correctness for all inputs.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m vectorpro --program results/request_cli/program.json --request experiments/requests/learn_xor.json
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m vectorpro --program results/request_cli/program.json --request experiments/requests/run_xor.json
```

The first request teaches a new capability and persists it in the single program
file on success. The second supplies only a name, numeric inputs and width and
executes the stored function. A known name bypasses learning. After installation,
`vectorpro` is also available as a console command. Exit status 0 means executed,
2 means missing evidence or learning failure, and 1 means an explicit error.
Program data is not overwritten by a failed request. This CLI is a structured
interface for a later LLM adapter; it does not interpret natural language.

For file/buffer requests, supply `--host-root` and encode operands as
`{"utf8": "relative/path"}` or `{"hex": "000102"}`. The CLI creates fresh handles
in that host context, rather than storing machine-specific handles in the file.
`"output_format": "hex"` exposes returned buffer bytes. Supported operations
remain the host primitives described above.

## Initial model: learning stateful procedures

The initial-model contract is in `docs/INITIAL_MODEL.md`. `StateLesson` describes
input types, initial file snapshots, exact target file snapshots, and optional
numeric return values. Optional `operations` traces list observed host effects
when final files alone do not distinguish behaviours. These are demonstrations,
not instruction recipes or argument routing. The bounded learner
searches generic typed host calls and argument routing, executes candidates only
in fresh `MemoryHostContext` instances, then checks separate validation cases.
An accepted procedure is an ordinary `VectorProgram` in the same saved file.

JSON requests can carry `state_lesson`; file contents are hex strings. The
example `experiments/requests/learn_transfer.json` teaches a transfer procedure
from state changes. It contains no read/write instruction recipe. Native
execution still requires `--host-root` and fresh path-buffer inputs. For example,
after preparing `input.bin` in a chosen host root:

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m vectorpro --program results/stateful_model/program.json --request experiments/requests/learn_transfer.json --host-root results/stateful_model
```

The current state search learns straight-line procedures within explicit step
and candidate budgets. With `control_flow: true`, it also searches a zero/nonzero
guard or one while region with numeric result feedback. `max_steps` counts calls;
guards add tensor steps, and every variant counts toward the candidate budget.
`time_budget_seconds` defaults to 60 and limits total state learning time. The
deadline is checked between candidates/cases and before registration. An active
candidate is not preempted; its existing kernel clock budget still applies.
The kernel clock budget rejects loops that do not halt. Nested/multiple control
regions and general mutable-variable routing remain outside this grammar.
Its primitive type catalog is provided execution/search machinery. CPU-trace
ingestion is not implemented. There is no task-name dispatch in the learner: the same search
also learns how to observe a byte without changing files. State search can now
call learned numeric units/programs and previously learned state procedures with
`input_types` and `output_type` contracts. Numeric operands are typed as values;
path/buffer handles are not passed as numeric arguments. Results fit the machine's
W-bit registers, including truncation of a numeric capability's wider output.
Older effectful programs without full type contracts remain executable but are
not inferred as search operators. Arithmetic learning
continues to use the existing unit/composition/bit-fold strategies. The optional
LLM adapter prepares intent/evidence; acquired functions do not depend on it.

`experiments/stateful_numeric_learning.py` learns XOR from JSON examples, learns
a first-byte observation procedure, then discovers a procedure combining both
from state examples. Its instruction sequence is not authored. The demo checks
native execution, reload, and 100 held-out file/operand cases.

`experiments/stateful_control_learning.py` discovers conditional file copying
from file-state examples and a variable-iteration read loop using an actually
learned numeric step. Repeated reads leave the same final files, so this second
lesson also supplies demonstrated host-effect traces. Neither lesson supplies
branch targets, loop boundaries or register routes. Each procedure is checked
on 100 held-out cases, native files and reload. This demonstrates the bounded
control grammar, not arbitrary file-processing loops.

The JSON example `experiments/requests/learn_conditional_transfer.json` uses
the same `state_lesson` entry point with `control_flow: true`. It can also reuse
an existing learned conditional procedure in the accumulated program file.

## Initial-model integration

`buffer_loops: true` enables a generic indexed-buffer fill/map search. The learner
searches numeric bodies and source/destination routes using acquired functions;
the iterator is an explicit supplied grammar, like BitFold. An acquired decrement
is discovered by behaviour probes and called by vector address at runtime.
`execution_budget` sets a persisted clock limit for larger files. This is bounded
program synthesis, not arbitrary control-structure discovery.

`experiments/initial_model.py` reuses previously learned sub/XOR tables and learns
byte fill, XOR mapping, a conditional call to the acquired map loop and a sequence
of two acquired loops. It checks 100 held-out cases per procedure, 2,048-byte
files, native execution and reload. No observed effect trace is needed for these
lessons: final file states distinguish the required mutations.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/initial_model.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python scripts/check_portability.py --program results/initial_model/program.json --output results/initial_model/portability-linux.json
```

The optional LLM adapter reads a tool guide and function schemas, looks up acquired
capabilities, proposes data-only lessons, teaches in memory, executes known user
requests, and asks for missing goals. Its bounded tool loop and Chat Completions
HTTP transport accept a user-configured model/endpoint; no model download or new
dependency is required. Acquired functions remain usable with the ordinary JSON
CLI after the model is disconnected. See [LLM adapter guide](docs/LLM_ADAPTER.md).
Protocol tests use scripted replies and a local HTTP fixture; live model quality
and Windows/macOS execution are not inferred from those tests.

On 2026-10-04 the user explicitly authorized native Windows tests in a desktop
directory. The same program SHA-256 passed the portability checker on Linux and
Windows without retraining; see `results/initial_model/portability-windows.json`.
An assistant also intercepted real tool calls to teach a new NAND function,
execute it after reload on 100 held-out 16/32-bit cases, and mutate a native file.
This exercised the assistant-to-learner bridge without an external model API.
macOS remains unverified. The default one-container test rule is retained.

## Tensor catalog and retrieval experiments

Path binding now copies quoted strings and bare filename tokens from the original
request, using the catalog's path types. Unicode names, quoted spaces and nested
paths are preserved. Filename digits are excluded from numeric operand choices;
missing paths lead to clarification. The backend rejects invented paths before
file access. Choosing the correct role among several mentioned paths is still a
semantic problem. See docs/LLM_ADAPTER.md for syntax limits and evidence-source gaps.
Full regression: 186 passed. New path evaluation: 4/6 overall, with exact target
paths in all five execution cases and clarification for the missing-path case.
Two Korean fills swapped width and byte value, incorrectly changing the test target
files while preserving other files. Path copying is not proof of parameter roles.
The reused NAND diagnostic still chose a buffer for numeric inputs and did not teach.
Raw successes/failures and source/data hashes are preserved in results/gemma_paths.

The follow-up catalog adapter isolates actual-input extraction, restricts numeric
bindings to literals in the original request, and verifies supplied caller evidence
without model transcription. A successful catalog request returns the executor's
`executed` receipt (outputs and exact request bindings) immediately, with no further
model turn. CLI treats it as success. Self-generated answers remain a separate problem.
New fixed binding tests also score width, operands and terminal status; those expected
bindings are never supplied to the model. See docs/LLM_ADAPTER.md.
Full regression: 183 passed. New first-use binding evaluation: 7/8 with caller
evidence, including exact width/operands and terminal-status checks. One Korean
filename was misread as IN; execution failed with original files unchanged.
This suite does not evaluate autonomous teaching, and is not directly comparable
with the earlier 7/12 output-based replay. Results are in results/gemma_binding.

The current local adapter uses Gemma 3 1B Instruct Q8_0 in the existing container.
Qwen weights were removed at the user's request; earlier results below are preserved.
Run `experiments/setup_gemma_model.py` inside `vectorpro-test` to restore the pinned
Gemma download with checksum verification. The native Gemma dialogue template uses
explicit JSON tool schemas; model decisions are still generated by real inference.
MiniLM retrieval and acquired vector programs are unchanged. See docs/LLM_ADAPTER.md.
Earlier Gemma replay passed 7/12 under the existing output/file checks; final conversation
and exact width interpretation remain imperfect. Both file transformations and
self-evidence/new learning failed; failed files were unchanged and no NAND was
registered. Full regression passed 181 tests. Raw results are in results/gemma_model.

The new prototypes store an exact typed tensor tree and semantic search vectors
in one `program.pt` file. Metadata can be reconstructed as JSON, edited and saved
back into the same file. The ordinary runtime JSON format remains unchanged.

`experiments/catalog_retrieval.py` replaces Qwen hidden-state pooling with a
trained multilingual MiniLM sentence encoder. Search documents come from tensor
calls, routing and control plus supplied machine semantics. An explicit input-type
filter removes incompatible candidates. It does not infer types from test labels.

Original requests improved from top1 1/4 to 4/4, with all four selected functions
executed correctly. On a separate final set of 24 English/Korean requests, semantic
search alone scored top1 12/24 and top3 21/24; supplied input types raised top1 to
24/24. Sixteen cases had one compatible candidate and eight required semantic
disambiguation. These are finite prototype results, not universal accuracy.

The actual Qwen tool-loop test passed known arithmetic, clarification and native
file execution under constrained decoding. New-function example generation still
failed; the later evidence-size guard rejected duplicate examples and asked for
input. The optional catalog adapter now connects search, required typed arguments,
isolated evidence verification and a single execution of the bound request. The
runtime supports `.pt` alongside JSON; known named functions still run without an
LLM or encoder. See [catalog adapter](docs/CATALOG_AGENT.md) for its evidence
requirements and remaining intent/example-generation limitations.
The evidence precheck and compact-example adapter initially passed 178 regression tests.
First-use Qwen evaluation passed 9/12: four numeric, two file, and three unsupported
requests with caller evidence. Self-generated XOR/NAND evidence failed. An ambiguous
file request incorrectly authorized a no-op using invented empty snapshots; files
were unchanged. The catalog LLM path now requires caller `state_validation` for
state execution. Rechecking that same failure passed 1/1; this is a regression,
not a new independent accuracy measurement. See docs/CATALOG_AGENT.md.
Final regression: 179 passed in 247.57 seconds; focused checks: 27 passed in 8.85 seconds.
Structured caller evidence can reject unsupported goals in isolated memory before
binding native inputs. Self-generated 4-bit examples use a separate model context
with actual bindings replaced by symbols; no candidate computes the answers.
`check_small` and `teach_small` retain backend limits and disjoint example checks.
Real Qwen development tests passed self-evidence XOR and unsupported deletion;
new NAND example generation still failed. Fixed first-use evaluation cases are
in `experiments/requests/catalog_agent_holdout_v2.json`.

Earlier real Qwen diagnostics passed 5/8;
after further phase/schema constraints, the native file request passed while
self-generated validation examples still failed. Those reused diagnostic sets
are reported separately, not as independent accuracy.

For prerequisites, commands, expected results and preserved output paths, see
[testing instructions](docs/TESTING.md), [tensor catalog details](docs/TENSOR_CATALOG.md)
and [LLM adapter results](docs/LLM_ADAPTER.md). Models live in the existing container,
outside the repository. macOS verification is deferred at the user's request.

## Roadmap

| | milestone |
|---|---|
| M0 | core abstractions + addition reproduction ✅ |
| M1 | sub, compare, bitwise, multiply at an equal data budget ✅ |
| M1b | four arithmetic operations + random expressions from four verified units ✅ |
| M2 | learning loop: plans → reuse/learn → searchable, explainable registry ✅ |
| M2b | all logic in vector programs: address-vector calls, routing and control tensors, task-agnostic kernel ✅ |
| M2c | learn loop programs from examples: generic bit fold, wider composition search ✅ |
| M2d | four arithmetic operations from examples: headroom, companion loops, boundary examples ✅ |
| M2e | LLM-written plans (requirements → curricula and target sources); nested loops; faster search |
| M3 | structure discovery beyond a fixed candidate set; capability search by behaviour vectors |
| M4 | ingest binaries via emulator traces (Unicorn), held-out compilers |
| M5 | synthesizer, effect log for native calls, cost vs native/interpreter |
