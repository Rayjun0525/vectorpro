# Initial model v0

Contract drafts now use isolated acquisition and acceptance against supplied examples,
input/output shape and permitted effects. See [CONTRACT_LEARNING.md](CONTRACT_LEARNING.md).
Registration does not independently certify the user's intent or model-generated labels.

The product is a learnable, portable vector program, with one request entry
point. A runtime supplies execution semantics; learned task logic is stored as
tables and program tensors, never Python source. The LLM is an optional intent
and teaching adapter, not a requirement for running an acquired function.

## Contract

Known requests execute the stored capability. Unknown requests require evidence
of the intended behaviour and enter bounded learning. A failed search returns
failure, not a guessed result. Accepted functions accumulate in the same file.

For calculations, evidence is input/output examples. For stateful work, evidence
is inputs, initial files, desired final files, optional numeric return value,
and optionally observed host-operation names in execution order. Such traces
describe demonstrated effects, not argument routing or a supplied program.
The target file snapshot is exact: unintended changes or extra files fail.
Training and validation cases are separate. Search uses training cases; a
candidate is registered only after it also passes supplied validation cases.
Independent held-out cases measure generalization and do not guide search.

## Execution and learning boundary

Structured input buffers use hexadecimal evidence.
State lessons may explicitly request `output_type: "buffer"` with hexadecimal
output targets, compared as actual bytes rather than transient handles. Portable
binary records currently combine one buffer and one unsigned value. Generic
packing/extraction is supplied; task routes are acquired. See [RECORD_RESULTS.md](RECORD_RESULTS.md).

Optional `list_loops` supplies a reverse list iterator with a searched typed argument-expression body and one
terminal value/effect call. This is a bounded grammar, not unrestricted list
program discovery. See [STRUCTURED_DATA.md](STRUCTURED_DATA.md).

The runtime provides generic buffer/file primitives and their input/output
types, analogous to a machine's instruction semantics. It does not provide task
recipes. The stateful learner enumerates typed call sequences and argument
routing, simulates them in a fresh memory filesystem, and compiles the successful
sequence to the existing vector instruction format. Its search library includes
learned numeric capabilities and previously learned state procedures with stored
type contracts. Numeric calls receive values, not path/buffer handles; their
results use the vector machine's W-bit register fitting semantics. Real execution uses fresh
OS data, not saved example outputs. Learning never accesses the native filesystem.

Stateful search defaults to straight-line sequences. With `control_flow: true`,
it also enumerates one zero/nonzero guard over a contiguous region or one while
region with numeric result feedback into an existing numeric operand. The guard,
region, feedback routing, polarity and calls are searched, not task recipes.
Zero iterations are supported and the existing kernel clock budget rejects
nonterminating candidates. `max_steps` counts calls; guards add tensor steps.
`time_budget_seconds` (default 60) bounds the complete state search using a
monotonic deadline, checked between candidates and cases and before acceptance.
An in-progress candidate finishes within its kernel clock limit before the
deadline is observed; this is not preemption of a host call.
Nested/multiple control regions and general mutable-variable routing are not
searched directly. Acquired procedures can be composed: a branch can call a
learned loop, and a sequence can call multiple independently learned loops.
With `buffer_loops: true`, an indexed-buffer grammar searches fill/map bodies,
source/target routing and acquired numeric operators. A stepping operator is
found by behaviour probes for decrement, like the primitive discovery in
BitFold; actual iteration and byte transformations run as tensor calls.
The grammar is explicitly supplied machinery, not unconstrained structure
discovery. `execution_budget` optionally persists a clock limit for larger
loops; failure to halt still raises an error after reload.
Arithmetic also has composition and bit-fold loop search. Arbitrary
data structures, CPU traces, and natural-language interpretation are future
extensions; their absence must not be hidden by task-
specific Python implementations. Human-designed low-level semantics and search
grammar are explicit; individual task procedures must be discovered.

## Initial proof of the concept

The user's first completion target is a tensor program that can use Linux:
filesystem, structured values and iteration, processes/stdio/pipes, networking,
and evidence-backed acquisition. See [LINUX_TARGET.md](LINUX_TARGET.md) for
acceptance criteria and measured progress. Filesystem host primitives now include
opt-in existence, move/delete, and directory creation/listing/removal. Stateful
evidence compares both exact files and directory state in isolated memory.
These semantics are supplied; task procedures are acquired as tensors.

Given example initial/final files, discover a file-transfer procedure without
providing calls, order, or argument routing. Run the learned tensor program on
unseen paths, empty/binary/longer content, and native files. Save and reload the
same program file and repeat. Negative controls must show that unsatisfiable
goals fail without native mutations or capability registration.

This is a bounded program-synthesis model, not a proof of replacing Rust or of
universal generalization. A found solution is correct on its checked cases.
The same learning interface should later receive evidence from an LLM, human
demonstration, or an instrumented existing program.

## Common function contracts

Acquired functions export versioned contracts from their stored types, tensors and
dependencies. Contracts live in the same program file; legacy files derive them on
load. Program callers use an exact content ID and named typed inputs without an
LLM, encoder, similarity search or implicit teaching. The optional catalog adapter
uses a compact projection of the same contract and retains evidence-gated execution.
Contract generation/export is not acquisition of a new function or a correctness
proof. See `docs/FUNCTION_CONTRACTS.md` for the v1 boundary and direct-call API.

## Optional LLM adapter and portability

`vectorpro.agent` supplies function schemas/instructions, a bounded conversation
loop and an HTTP Chat Completions transport with explicit endpoint/model.
It exposes capability lookup, data-only isolated teaching, known execution and
clarification. The adapter cannot submit Python code or tensor instructions.
Model-proposed examples are recorded as such and do not establish that the
human's intended result is correct. Tests use scripted model replies and a local
HTTP fixture: this verifies the integration protocol, not live-model quality.

`scripts/check_portability.py` executes the same saved initial-model file without
learning, checking Unicode paths, numeric functions, mutation and composed loops.
Linux and Windows results are recorded with identical program SHA-256. The
user explicitly authorized the Windows desktop test directory on 2026-10-04.
macOS remains unverified. Native OS results must not be inferred from source alone.
