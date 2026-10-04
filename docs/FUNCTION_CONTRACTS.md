# 공통 기능 계약과 직접 호출

초안에서 학습·등록하는 경로는 [CONTRACT_LEARNING.md](CONTRACT_LEARNING.md)를 참고한다.

프로그램과 LLM 어댑터가 같은 기능 계약을 사용한다. 계약은 저장된 기능의 타입,
셀·호출·라우팅·제어 텐서와 의존 기능에서 추출한다. 새 계약 초안만으로 기능을
등록하거나 학습 성공으로 간주하지 않는다. 배운 실행 본체가 있어야 호출 가능하다.
작업 알고리즘은 기존 벡터 프로그램에 있고 계약 계층은 조회·검증·호출만 수행한다.

## 조회와 직접 호출

```python
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime

runtime = VectorRuntime.load("program.pt", host=HostContext("work"))
contract = runtime.contract("map")
result = runtime.call_contract(
    contract["id"], {"x0": "자료.bin", "x1": 13}, width=24, version=1)
```

예시는 이미 배운 바이트 XOR 변환을 호출한다. 이름은 조회용이며 실제 호출은
계약 ID의 정확한 일치로 실행한다. 유사도 검색·LLM·인코더·학습기는 필요 없다.
`runtime.contracts()`는 전체 계약을 반환한다. 인자 키는 안정적인 슬롯 이름
`x0`, `x1`, …이며 각 슬롯의 타입과 역할을 계약에서 읽는다. 아직 사용자 정의
의미 이름(path/mask 등), 임의 자료구조, 배치 호출을 지원하는 계약은 아니다.

입력은 이름 있는 JSON 객체다. value는 unsigned integer, path는 호스트 루트
기준 상대 경로 문자열, buffer는 공백 없는 hex byte-pair 문자열이다. 원시 OS
핸들은 받지 않는다. width는 1..64의 정수이며 bool은 받지 않는다.
입력값은 width 범위에 맞아야 한다. 누락·추가 인자·잘못된 타입/범위·경로 이탈·
알 수 없거나 오래된 ID·계약 버전 오류는 실행 전에 예외로 반환한다.
모든 입력 검사를 마친 뒤에만 새 버퍼 핸들을 할당한다.

반환값은 `status`, `contract_id`, `contract_version`, `capability`, `outputs`,
`effects`다. 숫자 출력은 정수, 버퍼 출력은 `{hex: ...}`, 경로 출력은 `{utf8: ...}`다.
effects는 이번 호출의 호스트 작업 기록이며 내부 핸들을 포함할 수 있는 진단 정보다.
프로그램 파일에 실제 호스트 루트나 실행 핸들을 저장하지 않는다.
실행 도중 파일 오류나 예산 초과가 발생할 때 native 변경 전체를 롤백하는
트랜잭션 API는 아니다. 사전 계약 검증과 실행 중 오류를 구분한다.

## 계약 내용과 ID

계약 v1은 이름/설명, parameters(슬롯·타입·역할·JSON schema), 출력 타입/폭,
실행 주소 벡터/의존 실행 digest/호스트 작업 집합, 저장된 학습·검증 기록을 갖는다.
ID는 `vp1:<SHA-256>`이다. 계약 데이터와 실행 의존 내용의 지문이므로 재로딩과
무관한 기능 추가에서는 유지되고 실행 내용/의존 기능/계약 메타정보가 바뀌면 달라진다.
동등한 의미로 별도 학습한 기능이 반드시 같은 ID를 갖는 체계는 아니다.
주소는 내부 텐서 경로를 활성화하는 정보이며 호출자는 ID만 전달하면 된다.
version은 계약 형식 버전이고 ID는 저장된 특정 구현을 가리킨다.
SHA는 내용 식별과 변경 감지이며 파일의 인증 서명이나 범용 정확성 증명이 아니다.

계약은 JSON 저장의 `contracts` 필드와 `.pt`의 typed-tree 텐서에 함께 들어간다.
별도의 메타 JSON 파일을 읽어야 실행되는 구조가 아니다. 예전 저장 파일에 이 필드가
없으면 실행 데이터에서 계약을 추출한다. 새 저장 파일은 로딩 때 저장 계약과 실행
데이터에서 추출한 계약을 비교하고 불일치하면 거절한다. 기존 `request` API는 유지한다.
타입 정보가 없는 과거 효과 프로그램은 output type을 unknown으로 표시하고
새 API에서 실행을 거절한다. 기존 API로는 이전처럼 실행 가능하다.

검증 기록은 기존 학습/검증 이력이며 사용자의 의도나 모든 입력에 대한 독립 증명이
아니다. 런타임이 제공한 primitive는 학습된 기능과 provenance로 구분한다.

## LLM과 CLI

`vectorpro.contracts.tool_schema(contract)`는 같은 계약의 도구 JSON schema를
생성한다. 도구 이름은 ID 지문으로 만들며 외부 어댑터는 이 이름과 ID를 매핑해
`call_contract`로 전달할 수 있다. 자동 도구 등록/원격 서버는 이번 구현에 포함하지 않는다.
카탈로그는 같은 계약의 ID/version/타입/역할만 압축해 작은 모델에게 제공한다.
전체 실행 주소와 학습 이력을 매 검색 프롬프트에 넣지 않는다.
기존 카탈로그 LLM의 단일 실행은 근거 검증 후 공통 API로 연결한다. 직접 호출
API를 미검증 LLM 도구로 공개해 기존 승인 경계를 우회하지 않는다.
카탈로그가 없는 예전 경로와 여러 숫자 lane 호출은 기존 API를 유지한다.

CLI 조회 요청:

```json
{"action":"contracts","name":"map"}
```

CLI 실행 요청(ID는 조회 결과를 사용):

```json
{"action":"call_contract","contract_id":"vp1:<조회한 ID>","version":1,
 "width":24,"arguments":{"x0":"자료.bin","x1":13}}
```

```powershell
nerdctl exec vectorpro-test python -m vectorpro --program results/contracts_model/program.pt --request request.json --host-root work
```

조회/직접 호출 CLI는 프로그램을 다시 저장하지 않는다. contract action에 legacy
operands/lesson 등을 섞으면 거절한다. API 예외는 CLI의 JSON error/종료 코드1로
반환한다. unknown ID는 자동 학습이나 가까운 기능 실행으로 전환하지 않는다.

## 검증과 남은 범위

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_contracts.py tests/test_semantic_catalog.py tests/test_agent.py tests/test_runtime.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/contract_calling.py
```

데모는 기존 `results/contracts_model`을 덮어쓰지 않는다. 반복할 때 `--output`을
새 이름으로 정한다. 기존 학습 파일을 계약 포함 `.pt`로 저장하고 서로 다른 두
native 루트에서 같은 ID로 숫자/파일 변환/쓰기와 실패 입력 무변경을 검증한다.
LLM/transformers/llama_cpp는 import하지 않는다. 보고서는 관찰 결과이며
실행에 필요한 별도 계약 파일이 아니다. JSON/PT 호환, ID와 의존 기능 변경,
잘못된 입력·symlink 이탈 거절, portable buffer 출력, CLI, 공통 어댑터 호출을
테스트한다. 새 기능 학습이나 새로운 OS 이식성 시험은 아니다.

최종 관련 56 passed, 24.59초. 전체 회귀 207 passed, 285.05초이며 전체 수집 뒤
추가된 계약 테스트 2건은 관련 범위에서 별도로 통과했다. 실제 Gemma 기존 요청
회귀는 2/2로 독립 첫 평가가 아니다. 모델 추론과 회귀를 병행해 시간은 속도
비교에 쓰지 않는다. 결과/SHA/한계는 `results/contracts_model/verification.json`에 있다.

현재는 공통 기능 계약과 초안의 검증·학습·등록 경로까지 연결했다. 실제 Gemma의
초안 작성 평가, 숫자/경로 역할과 입력 타입의 정확한 자연어 해석, 독립 근거를 통한 자동
학습, 더 넓은 자료구조/합성 계약이 남아 있다. macOS는 보류한다.
