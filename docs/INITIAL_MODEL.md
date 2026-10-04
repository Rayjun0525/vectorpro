# Initial model v0

The product is a learnable, portable vector program, with one request entry
point. A runtime supplies execution semantics; learned task logic is stored as
tables and program tensors, never Python source. The LLM is an optional intent
and teaching adapter, not a requirement for running an acquired function.

## Contract

Known requests execute the stored capability. Unknown requests require evidence
of the intended behaviour and enter bounded learning. A failed search returns
failure, not a guessed result. Accepted functions accumulate in the same file.

For calculations, evidence is input/output examples. For stateful work, evidence
is inputs, initial files, desired final files, and optional numeric return value.
The target file snapshot is exact: unintended changes or extra files fail.
Training and validation cases are separate. Search uses training cases; a
candidate is registered only after it also passes supplied validation cases.
Independent held-out cases measure generalization and do not guide search.

## Execution and learning boundary

The runtime provides generic buffer/file primitives and their input/output
types, analogous to a machine's instruction semantics. It does not provide task
recipes. The stateful learner enumerates typed call sequences and argument
routing, simulates them in a fresh memory filesystem, and compiles the successful
sequence to the existing vector instruction format. Real execution uses fresh
OS data, not saved example outputs. Learning never accesses the native filesystem.

The initial stateful search is deliberately bounded to straight-line sequences.
Arithmetic already has composition and bit-fold loop search. Stateful branches,
variable-length loops, arbitrary data structures, CPU traces, and natural-language
interpretation are future extensions; their absence must not be hidden by task-
specific Python implementations. Human-designed low-level semantics and search
grammar are explicit; individual task procedures must be discovered.

## Initial proof of the concept

Given example initial/final files, discover a file-transfer procedure without
providing calls, order, or argument routing. Run the learned tensor program on
unseen paths, empty/binary/longer content, and native files. Save and reload the
same program file and repeat. Negative controls must show that unsatisfiable
goals fail without native mutations or capability registration.

This is a bounded program-synthesis model, not a proof of replacing Rust or of
universal generalization. A found solution is correct on its checked cases.
The same learning interface should later receive evidence from an LLM, human
demonstration, or an instrumented existing program.
