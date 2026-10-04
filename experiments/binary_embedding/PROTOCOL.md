# Side experiment: Rust binary embeddings

This experiment is separate from vectorpro's milestones. The protocol is fixed
before collecting results. It does not modify the main learner or kernel.

## Question and scope

Do raw function-code embeddings retain enough information to distinguish four
arithmetic behaviours on held-out source transformations? Can a supervised
decoder also predict their finite output tables? Neither result establishes
general machine-code execution, arithmetic-rule discovery, or input-width
generalization.

## Corpus

- Four operations on unsigned 8-bit values: wrapping add, wrapping subtract,
  wrapping multiply, and unsigned division (zero divisor returns 255).
- Eight reversible wrapper templates, four seeded constant variants per
  template, and all four operations per variant: 128 compiled functions.
- Wrappers scramble and restore operands and results. `black_box` prevents
  their wholesale removal. They do not change the arithmetic behaviour.
- Training: templates 0–3, optimization level 1, 64 functions.
- Held-out evaluation: templates 4–7, optimization level 3, 64 functions.
  Template and optimization changes are combined; this cannot isolate their
  individual effects. Compiler, architecture and OS remain fixed.
- Every function has the same symbol `f`. Features use ONLY `.text.f` bytes:
  no source, labels, file paths, symbols, ELF headers, or disassembly.
- Reject exact code duplicates between train and test. Check each native
  function on all 65,536 uint8 input pairs against independent references.

## Models fixed before execution

- Main embedding: normalized 256-bin byte histogram plus normalized 1,024-bin
  hashed adjacent-byte histogram, concatenated and L2 normalized.
- Main model: centered kernel ridge regression with alpha=0.01, no tuning.
- Control: the same regression using standardized log code length only.
- Primary endpoint: four-class accuracy on the 64 held-out binaries; chance
  under the balanced label null is 25%. Report confusion matrix and per-template
  accuracy.
- Secondary endpoint: predict the 256-entry output table for inputs 0..15
  from the embedding using a separate eight-bit-output ridge decoder. All
  table targets are obtained by executing training binaries. Test tables are
  withheld. No handwritten arithmetic operation is selected by the decoder.
  Report exact output accuracy, bit accuracy, and fully correct binary tables.
  This tests transfer to new binaries, NOT transfer to unseen numeric inputs.
- Save the embedding/regression tensors and held-out predictions as NPZ.

## Statistics and controls

- Primary null: within each held-out wrapper-template block, operator labels
  have no association with predictions. Apply the same permutation to all four
  constant variants in a template. Enumerate all 24^4=331,776 permutations via
  convolution of the four block count distributions. Report exact one-sided p.
  This preserves dependence among same-template variants; examples and input
  pairs are NOT treated as independent evidence.
- Report a 95% percentile bootstrap interval by resampling the four held-out
  template blocks, 20,000 times. Four blocks give limited population inference.
- Compare against length-only control; report both, without claiming an
  improvement solely from separate p-values.
- Negative control: 1,000 ridge classifiers with operator labels permuted
  independently within training families; report held-out accuracy distribution.
- Secondary measures are exploratory; the primary endpoint is fixed and no
  hyperparameters or templates are selected using test performance.

## Reproduction

Use the existing `vectorpro-test` container ONLY. Install `rustc`, `gcc` and
`binutils` in that container; do not build images or create other containers.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 vectorpro-test python experiments/binary_embedding/run.py
```

Temporary native artifacts stay under `/tmp/vectorpro-binary-embedding` in the
same container. Results go to `results/binary_embedding/` in the workspace.
Rust sources are saved there for audit, separately from model features.

Interpretation must remain conditional on this synthetic corpus and the fixed
toolchain. Short opcode patterns and code length may explain success. This
experiment does not establish that four binaries suffice to learn Rust or an
OS, or that the saved tensors replace the existing vector execution kernel.
