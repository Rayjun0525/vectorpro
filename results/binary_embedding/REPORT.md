# Binary embedding side experiment — 2026-10-04

## Finding

Raw function bytes contain a statistically detectable signal for distinguishing
the four trained arithmetic behaviours in this controlled corpus. The saved
learned decoder also transfers finite output tables to some new binaries.
It does not establish that embeddings have learned to execute machine code.

## Design

Rust 1.85.1 / LLVM 19.1.7, x86-64 Linux, in the existing `vectorpro-test`
container. No new image was built and no additional container was created.
Compiler and dependencies remain installed in that container for reuse.

Compiled 128 functions: 4 operations × 8 reversible wrapper templates × 4
constant variants. All functions have the same symbol `f`. Wrappers preserve
the result while varying code bytes. Arithmetic is unsigned 8-bit wrapping;
division by zero returns 255.

64 functions from templates 0–3 at optimization level 1 were used for training.
64 from previously unseen templates 4–7 at optimization level 3 were held out.
There are no exact byte duplicates across the split (all 128 code sequences
are unique). Source templates and hyperparameters were fixed before execution.
Rust's documented object emission and optimization options were used:
[rustc command-line arguments](https://doc.rust-lang.org/rustc/command-line-arguments.html)
and [codegen options](https://doc.rust-lang.org/rustc/codegen-options/).

ONLY the raw `.text.f` section was embedded. Source text, operation labels,
file names, symbols, ELF headers, and disassembly are not model features.
The feature encoder is fixed: byte frequency plus hashed adjacent-byte
frequency, totaling 1,280 dimensions. Centered ridge regression learns the
decoder weights; alpha=0.01, with no test-set tuning. This is a lightweight
baseline, not a learned neural byte encoder.

Each native function was checked on every uint8 input pair, for 8,388,608
reference comparisons, all exact. These comparisons establish corpus
correctness; they are not successes by the learned predictor.

## Primary endpoint: operation identification

| Model | Held-out correct | Accuracy |
|---|---:|---:|
| Byte/bigram embedding + learned classifier | 52 / 64 | 81.25% |
| Log byte length + same regression method | 16 / 64 | 25.00% |
| Shuffled training labels, mean over 1,000 fits | — | 24.38% |

The balanced label-null chance rate is 25%. The length control predicts
division for every held-out sample: it is a specified simple baseline, not
proof that all length-based or opcode-based shortcuts are eliminated.

Confusion matrix (rows are actual operation, columns are prediction):

| Actual / predicted | add | sub | mul | div |
|---|---:|---:|---:|---:|
| add | 12 | 3 | 1 | 0 |
| sub | 0 | 15 | 1 | 0 |
| mul | 0 | 0 | 16 | 0 |
| div | 0 | 0 | 7 | 9 |

Accuracy by held-out template is 81.25%, 75.00%, 100.00%, and 68.75%.

### Statistical interpretation

For each held-out template, permute the four operation labels consistently
across its four constant variants. There are 24 permutations per template,
and 24^4 = 331,776 joint permutations. Exactly 4 give at least 52 correct:

`p = 4 / 331,776 = 0.0000120563` (one-sided).

This exact conditional test finds an association between predictions and
operation labels under the specified block-permutation null. It is not the
probability that the model is wrong or a proof of semantic understanding.
The statistic is computed at binary level with template dependence preserved,
not by treating thousands of arithmetic inputs as independent samples.

A 20,000-resample template-block bootstrap gives a 95% percentile interval
of 71.875%–93.75%. Only four test templates are available, so the interval
provides limited evidence about performance on a wider population of programs.
Shuffled-training-label accuracies have a central 95% range of 9.375%–42.188%.

## Exploratory endpoint: direct output-table prediction

A second learned decoder maps the code embedding to the eight output bits
for each of 256 operand pairs (a,b in 0..15). Training targets come from native
execution. The decoder does not select or call handwritten arithmetic
functions. Held-out output tables are used for evaluation only.

| Metric | Byte embedding decoder | Length-only decoder |
|---|---:|---:|
| Exact numeric outputs | 12,558 / 16,384 (76.65%) | 4,800 / 16,384 (29.30%) |
| Output bit accuracy | 92.42% | 68.48% |
| Entire 256-entry tables correct | 43 / 64 | 16 / 64 |

The table targets contain only four distinct arithmetic behaviours repeated
across binaries. Transfer can therefore arise from identifying which behaviour
is present and reconstructing a familiar table. It does not show discovery
of an arithmetic algorithm. All numeric inputs evaluated here also occur
in the training tables; unseen-input and width generalization were not tested.

The saved tensor predictor was reloaded and reproduced all decoder predictions.
The standalone predictor was also run on the first four held-out binaries at
inputs (7,3), in their pre-existing order:

| Actual operation | Expected | Predicted |
|---|---:|---:|
| add | 10 | 10 |
| sub | 4 | 4 |
| mul | 21 | 21 |
| div | 2 | 21 |

The division failure is retained. This predictor is not reliable enough to
replace native execution.

## What this justifies next

The observed signal justifies a follow-up experiment on binary-conditioned
state transitions or output prediction for unseen operand pairs. It does not
justify claiming that four binaries are sufficient to learn Rust, that the
model is an OS-independent executor, or that the main vectorpro kernel has
acquired binary ingestion. A follow-up should also isolate optimization from
template changes and hold out a compiler or instruction architecture.

No main `src/vectorpro` code was modified. All additions are confined to this
side experiment and its result directory. The existing 101-test suite was
not rerun because its code was unchanged.

## Artifacts and reproduction

- `experiments/binary_embedding/PROTOCOL.md`: design fixed before execution.
- `experiments/binary_embedding/run.py`: corpus generation, training, statistics.
- `experiments/binary_embedding/predict.py`: saved tensor predictor.
- `results/binary_embedding/results.json`: measured results and environment.
- `results/binary_embedding/corpus.json`: code bytes, source, split and hashes.
- `results/binary_embedding/model_and_predictions.npz`: learned weights and
  evaluation arrays. Evaluation targets are not consumed by `predict.py`.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 vectorpro-test python experiments/binary_embedding/run.py
nerdctl exec -e OPENBLAS_NUM_THREADS=1 vectorpro-test python experiments/binary_embedding/predict.py --code /tmp/vectorpro-binary-embedding/sample_064.bin --a 7 --b 3
```
