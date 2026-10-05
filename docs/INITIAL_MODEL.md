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
sequence to the existing vector instruction format.
Acquired stdout-returning functions can themselves be composed to learn sequential
process input/output routing. `branches_only: true` with `control_flow: true`
restricts the general grammar to guards while preserving the old default branch/while
search. It supplies no condition or task recipe. See [PROCESS_CHAIN.md](PROCESS_CHAIN.md).

Its search library includes
learned numeric capabilities and previously learned state procedures with stored
type contracts. Numeric calls receive values, not path/buffer handles; their
results use the vector machine's W-bit register fitting semantics. Real execution uses fresh
OS data, not saved example outputs. Learning never accesses the native filesystem.
Process calls in memory use exact caller-supplied request/stdin/stdout/stderr/exit
observations, never launch native processes, and reject unrecorded calls. These
observations are evidence rather than independent correctness certificates.
See [PROCESS_RESULTS.md](PROCESS_RESULTS.md) for provided semantics and limits.

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

## 2026-10-05: 리눅스 기능 통합

스트리밍 파이프, 프로세스 전후 전체 상태 관찰, HTTP 및 시스템 조회와 조합 학습을 추가했다. 다섯 기능의 학습/검증과 저장 후 실제 리눅스 13건 검증을 통과했다. 실행 기능과 습득 절차의 구분, 재현 명령, 현재 범위와 남은 과제는 docs/LINUX_BUNDLE.md를 참조한다. 최종 증거는 results/linux_bundle_final_verified에 있다. 전체 회귀 결과는 docs/HANDOFF.md에 기록한다.

## 2026-10-05: 호출자 근거와 숨긴 사례를 통한 학습 채택

EvidenceBank 모드에서 LLM은 근거 ID와 새 이름을 선택하고, 백엔드가 고정된 인터페이스와 예제로 격리 학습한다. 학습에 제공하지 않은 사례를 통과하고 저장이 성공해야 등록한다. 승인 계약은 LLM 없이 직접 재사용할 수 있다. 실제 의도와 근거 출처를 자동 인증한 것은 아니다. 명세와 재현은 docs/VERIFIED_ACQUISITION.md, 최종 검증은 docs/HANDOFF.md를 참조한다.

## 2026-10-05: 설치된 외부 기준에서 근거 자동 수집

ReferenceProviders가 제공된 manifest의 실행 파일을 임시 루트에서 관찰해 학습/검증/숨긴 사례를 자동 생성한다. LLM은 기준 ID만 선택하고 실제 사용자 파일은 수집에 사용하지 않는다. 기준 설치와 의미 선택의 독립 검증은 여전히 남아 있다. 명세/재현은 docs/REFERENCE_EVIDENCE.md, 검증 결과는 docs/HANDOFF.md를 참조한다.

## 2026-10-05: 새 표현의 요청을 검증해 기존 계약 재사용

기준 관찰 뒤 기존 계약을 현재 근거 전체로 검사하고, 하나만 통과하면 재학습 없이 요청 연결을 저장한다. 기존 텐서/계약 ID는 유지하며 요청 연결은 같은 프로그램 파일에 포함한다. 여러 후보나 한도 초과는 승인하지 않는다. 명세와 재현은 docs/VERIFIED_REUSE.md, 결과는 docs/HANDOFF.md를 참조한다.

## 2026-10-05: 호출자 목표 검사

호출자가 제공한 원문 요청과 선언적 상태 조건을 기준의 전체 관찰 사례에 적용한다. 모순이면 학습·재사용·저장·실제 실행을 거절한다. 승인 기록에는 목표 해시를 저장하며 다른 목표나 목표 미검증 연결은 이 모드에서 실행하지 않는다. 사용 횟수 자체가 모델 정확도를 높이지는 않는다. 목표 자동 해석과 자연어 의도의 독립 검증은 남아 있다. 명세·API·CLI·재현은 docs/GOAL_EVIDENCE.md, 최종 결과는 docs/HANDOFF.md를 참조한다.

## 2026-10-05: 자연어 목표 초안과 검토

모델이 최종 상태 조건 초안을 만들면 한국어 검토 내용과 목표 해시를 반환한다. 호출자의 확인 전에는 관찰·학습·실행하지 않는다. 확인된 목표는 기존 목표 검사 경로에 전달한다. CLI --draft-goal, propose_goal/accept_goal API를 추가했다. Gemma 첫 초안 정확도는 1/3으로 아직 충분하지 않다. 명세·한계·재현은 docs/GOAL_DRAFT.md, 최종 결과는 docs/HANDOFF.md를 참조한다.

## 2026-10-05: Gemma 목표 작성 정확도 개선

인자별 최종 상태 선택, 백엔드의 전체 보존 조건 구성, 표기 예시와 지원 범위 판단의 문맥 분리를 적용했다. 기존 rules 방식도 유지한다. 기존 요청 재시험은 1/3에서 2/3이지만 암호화의 잘못된 초안 반환이 남았다. 새 영어·한국어 요청 8건은 별도로 고정해 이전 방식과 비교한다. 모든 중간 실패·방법·비용·재현·한계는 docs/GOAL_ACCURACY.md, 최종 결과는 docs/HANDOFF.md를 참조한다.

## 2026-10-05: 목표 예제의 벡터 검색으로 정확도 개선

Gemma 프롬프트 변경은 새 평가에서 안정적으로 개선되지 않아 기본 rules를 유지한다. 선택적 GoalMemory는 고정 목표 예제13개와 기존 multilingual MiniLM을 사용해 초안을 검색한다. 새 요청8건의 동일 비교는 Gemma2/8에서 검색6/8로 개선됐다. 이동1건의 오답과 보존1건의 보수적 거절이 남아 자동 승인하지 않는다. 예제·벡터는 같은 프로그램 파일에 저장한다. LLM 가중치를 학습한 결과는 아니다. 모든 방법/실패/재현/비용/한계는 docs/GOAL_ACCURACY.md, 최종 결과는 docs/HANDOFF.md를 참조한다. 전체 회귀375 passed (262.20초), 관련56 passed (4.46초).
