# 작업 인수인계

기록 기준: 2026-10-04, 한국 시간. 초기 기반은 `ad640f6`으로 원격 `main`에
반영됐다. 이후 숫자·상태 기능 통합, 제한된 분기·반복, 실제 파일 변환 반복,
조합 학습과 선택적인 LLM 어댑터를 구현했다. 최신 커밋·미커밋 상태는
`git status`와 `git log`로 확인한다. 관련 명세와 실험은 아래에 기록한다.

## 현재 구현

| 위치 | 역할 |
|---|---|
| `src/vectorpro/runtime.py` | 알려진 요청 실행, 미지 요청 학습, 단일 파일 저장·재로딩 |
| `src/vectorpro/agent.py` | 선택적인 LLM 함수 호출 어댑터, 학습·실행·조회·질문 |
| `src/vectorpro/__main__.py` | JSON 요청 CLI. `python -m vectorpro` 또는 설치된 `vectorpro` |
| `src/vectorpro/host.py` | 버퍼·파일 실행 기능, 격리된 메모리 파일 환경, 기본 타입 규약 |
| `src/vectorpro/learning/examples.py` | 정답 함수 없는 숫자 예제 `ExampleLesson`과 검증 |
| `src/vectorpro/learning/learner.py` | 기존 규칙·조합·bit-fold 학습과 유한 예제 학습 |
| `src/vectorpro/learning/stateful.py` | 파일 상태·시연에서 순차 절차 및 제한된 분기·반복 탐색 |
| `src/vectorpro/learning/buffer_loops.py` | 범용 인덱스 반복 문법에서 파일 변환 본문·인자 탐색 |
| `src/vectorpro/machine/` | 호출·분기·반복 실행. `calls` 텐서로 반환값 없는 호출 지원 |
| `src/vectorpro/learning/registry.py` | 기능 보관·설명·주소 해석·저장, 효과 있는 기능의 계산 탐색 제외 |

상태 학습은 입력 타입, 초기 파일, 정확한 최종 파일, 선택적 숫자 반환값을 받는다.
사용자는 호출 이름·순서·인자 연결을 제공하지 않는다. 후보를 가상 환경에서
실행하고 사례를 만족하는 순서를 일반 `VectorProgram`으로 저장한다.
기본 OS 동작과 타입 카탈로그는 실행 기반으로 사람이 제공한다.

## 숫자 기능과 상태 절차의 통합 탐색

상태 학습기의 탐색 라이브러리가 이제 배운 숫자 기능과 타입 계약을 가진 상태
절차도 포함한다. 숫자 기능은 실제 학습된 실행기를 호출하며 Python 산술로
대체하지 않는다. 숫자 인자는 `value`, 파일 경로와 버퍼는 별도 타입이다.
계산 결과는 벡터 머신의 W비트 레지스터에 맞춰 저장한다.

새 상태 기능은 `input_types`와 `output_type`을 함께 저장한다. 이전 효과 있는
프로그램에 반환 타입이 없으면 실행은 유지하되 탐색에서 타입을 임의로 추론하지
않는다. 기계의 인자 슬롯 수보다 큰 기능은 상태 탐색 후보에서 제외한다.

검증 실험 `experiments/stateful_numeric_learning.py`는 JSON 예제에서 숫자
기능을 학습하고, 파일의 첫 바이트를 관찰하는 절차를 학습한 뒤 두 기능을
호출하는 새 절차를 상태 예제만으로 발견한다. 작업 순서는 제공하지 않는다.
2단계 절차는 328개 후보에서 발견됐고, 별도 사례 100개·실제 파일·재로딩이
모두 일치했다. 결과는 `results/stateful_numeric_model/summary.json`에 있다.

새 관련 테스트 10개는 숫자 기능의 직접 호출, 배운 상태 절차 재사용,
숫자 기능이 없을 때의 실패, 기존 상태 학습의 회귀를 확인한다.

## 최근 검증

- 파일 변환 반복·조합·LLM 어댑터를 포함한 전체 **156 passed**, 201.40초.
- fill/map/guarded_map/pair_map 각각 미사용 **100/100**, 2048바이트·실제 파일·재로딩 성공.
- 동일 프로그램 파일의 Linux 이식성 검사 성공. 숫자·파일 변환·조건부/다중 반복·한글 경로 확인.
- LLM은 스크립트 모델 응답과 컨테이너 내 실제 HTTP 전송 fixture로 검증했다.
- 같은 컨테이너에 의존성 추가 없이 편집 설치를 갱신하고 `vectorpro-agent --help` 진입점을 확인했다.
- 전체 학습 시간 제한 보완 후 **148 passed**, 150.03초. 같은 컨테이너 재시작 후 실행.
- 시간 초과, 학습/검증 통과 후 마감 초과 미등록, 잘못된 한도 값 거절,
  이전 JSON 기본값 호환을 확인했다.
- 분기·반복 확장 최종 전체 회귀 **138 passed**, 205.06초. 기존 컨테이너만 사용.
- 분기·반복 각각 별도 사례 **100/100**, 실제 파일·재로딩·JSON CLI 성공.
- 비종료 효과 후보의 메모리 격리·한도 초과 거절 및 실패 후보 미등록 확인.
- 숫자·상태 기능 통합 후 같은 `vectorpro-test` 컨테이너에서 전체 **134 passed**, 167.81초.
- 통합 실험의 별도 사례 **100/100** 성공, 실제 파일 실행·저장 후 재실행 일치.
- 같은 `vectorpro-test` 컨테이너에서 전체 **131 passed**, 133.28초.
- 이후 상태 설명·메타데이터 보완을 포함한 관련 테스트 **19 passed**.
- 상태 예제 실험: 학습 2개·검증 2개로 43개 후보에서 2단계 파일 전달 절차 발견.
- 별도 미사용 사례 20개 모두 성공. 빈 내용·바이너리·긴 내용·새 경로 포함.
- 실제 파일 결과와 저장 후 재실행 결과 일치.
- 동일한 탐색기가 파일의 첫 바이트를 반환하는 다른 절차도 발견.
- 가상 탐색의 실제 파일 접근 금지, 실패 후보 미등록, 의도하지 않은 파일 변경
  거절, 실행 예산 제한을 테스트했다.
- 이전 M2d 레지스트리의 곱셈·나눗셈·나머지 로딩과 실행도 확인했다.

이 수치는 기록이며 다음 변경의 성공을 보장하지 않는다. 숫자·상태 학습의 사례
검증은 전체 입력 공간에 대한 증명이 아니다.

## 제한된 분기·반복 학습

`StateLesson.control_flow=True` 또는 JSON `control_flow: true`로 제어 후보를
추가한다. 기본값은 기존 순차 탐색을 유지한다. 일반 호출 후보의 연속 구간에
0/비0 조건을 삽입하거나, 숫자 호출 결과를 해당 호출의 기존 숫자 인자에
피드백하는 while 구간을 탐색한다. 구간·조건·인자 연결은 사람이 지정하지
않는다. 핸들·상수는 피드백 대상에서 제외한다. while에는 선행 조건 검사가
있어 0회 반복이 가능하다. 실행 커널과 저장 형식은 기존 것을 그대로 쓴다.

`max_steps`는 호출 개수이며 조건 명령은 추가 텐서 단계다. 제어 후보와 반환
경로 후보도 각각 candidate budget에 포함한다. 종료하지 않는 후보는 기존
커널의 clock budget 초과로 거절한다. 아직 한 프로그램에서 중첩 또는 여러
독립 제어 구간을 함께 찾지는 않는다.

상태 학습의 전체 시간은 `time_budget_seconds`로 제한한다(기본 60초).
후보·사례 사이 및 등록 직전에 단조 시계의 마감 시간을 확인하며, 한도 초과는
`time budget` 실패로 반환한다. 실행 중인 호출을 강제 중단하지는 않으므로
현재 후보의 커널 clock budget만큼 마감 확인이 지연될 수 있다.

`StateExample.operations`는 선택적인 호스트 동작 시연 기록이다. 실제 동작
이름과 순서가 전부 일치해야 하며, 빈 기록은 호스트 호출이 없어야 한다는
뜻이다. 미제공과 빈 기록을 구분한다. 최종 상태가 같은 반복 읽기의 경우
이 기록이 횟수를 구분하는 학습 근거다. CPU 트레이싱 기능은 아니다.

`experiments/stateful_control_learning.py`는 조건부 파일 복사와 배운 숫자
연산을 재사용하는 반복 읽기를 발견한다. 각 절차에서 별도 사례 100개,
실제 파일, 저장 후 재실행을 확인한다. 작업 프로그램은 작성하지 않는다.
이 데모의 반복 학습에는 시연 기록을 제공하며, 임의의 파일 처리 반복을
모두 학습했다는 의미는 아니다. 결과는 `results/stateful_control_model/`에 있다.
최종 데모는 조건부 복사를 1041개 후보, 반복 읽기를 147개 후보에서 찾았다.
조건부 복사의 JSON 요청은 `experiments/requests/learn_conditional_transfer.json`이다.
CLI로 이 요청을 기존 실험 프로그램에 추가했을 때 이미 배운 조건부 복사를
재사용하는 1단계 호출을 36개 후보에서 찾아 실제 파일에 실행했다.

## 재현

컨테이너 규칙은 루트 `AGENTS.md`를 따른다. Python, PyTorch, pytest, Rust,
gcc, binutils는 같은 컨테이너에 설치돼 있다. `/workspace`는 이 저장소다.

```powershell
nerdctl ps -a --filter name=vectorpro-test
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/stateful_learning.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/stateful_numeric_learning.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/stateful_control_learning.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/initial_model.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python scripts/check_portability.py --program results/initial_model/program.json --output results/initial_model/portability-linux.json
```

`experiments/stateful_learning.py`는 소유한 `results/stateful_model/` 데모 자료를
재생성한다. `summary.json`에 측정값과 발견한 절차가 있으며 `program.json`은
그 기능을 담은 단일 프로그램 파일이다. 이 자료를 사용자 운영 파일로 쓰지 않는다.

숫자 학습 CLI 예제는 `experiments/requests/learn_xor.json` 및 `run_xor.json`,
상태 학습 요청 예제는 `learn_transfer.json`이다. 새 숫자 학습을 재현할 때는
아직 해당 기능이 없는 프로그램 파일을 사용한다. 이미 있으면 학습을 건너뛴다.

## 한계와 다음 범위

- 상태 탐색은 기본적으로 순차 호출, 선택적으로 한 개의 조건 구간 또는
  숫자 피드백 while 구간을 지원한다. 인덱스 기반 버퍼 반복 문법도 추가됐다.
  배운 반복을 조건에서 호출하거나 여러 번 호출하는 조합은 가능하다.
  임의의 중첩 제어 구간·범용 변수 갱신·복잡한 자료구조의 탐색은 남아 있다.
- 산술은 기존 범용 구조 후보와 bit-fold 검색을 사용한다.
- LLM 도구 연결·HTTP 전송·조회·학습·실행·추가 질문 인터페이스는 구현됐다.
  검증은 스크립트 모델 응답/HTTP fixture로 수행했다. 실제 모델의 자연어
  해석 품질은 모델·주소·인증 설정과 별도 실측이 필요하다. CPU 추적은 미구현이다.
- Linux 컨테이너에서 검증했다. Windows/macOS 설치·동일 동작 검증은 남아 있다.
- 우선 초기모델을 확실히 한다. 초기모델 이후의 운영 기능을 성급히 확장하지 않는다.
- `ls`는 사용자가 가능성을 질문한 사례이지 현재 구현 과제가 아니다.

## 파일 변환 초기모델과 외부 연결

`StateLesson.buffer_loops=True`는 버퍼 길이에 따라 인덱스를 순회하는 범용
문법을 사용한다. 인덱스 감소에 필요한 연산은 학습된 숫자 기능에서 행동
probe로 찾고, 실제 반복의 연산은 텐서 호출로 실행한다. Python에 특정 작업의
바이트 변환을 구현하지 않는다. 입력/출력 파일 연결·변환 본문은 예제에서 찾는다.
`execution_budget`은 더 큰 반복을 위한 실행 clock 한도이며 저장 후에도 유지한다.

`experiments/initial_model.py`는 이전에 실제 학습된 sub/xor 테이블을 가져와
fill/map을 최종 파일 상태만으로 학습한다. 이 반복들을 조건에서 호출하는 기능과
두 파일에 각각 호출하는 기능도 학습한다. 시연 기록 없이 네 기능 각각 별도
100개 사례가 성공했고, 2048바이트·실제 파일·재로딩이 일치했다.
자료는 `results/initial_model/`에 있으며, task program은 사람이 작성하지 않았다.
다만 반복 문법은 사람이 제공한 범용 구조이므로 무제약 구조 발견은 아니다.

`VectorRuntime.teach`는 실제 입력에서 실행하지 않고 기능만 습득하는 경로다.
LLM 어댑터는 이 경로와 알려진 요청 실행을 따로 사용한다. 동작/설정/정답의
한계는 `docs/LLM_ADAPTER.md`, 플랫폼 검증은 `docs/PORTABILITY.md`에 있다.
실제 LLM endpoint/model은 아직 제공되지 않았다. 모든 테스트를 한 컨테이너에서
수행하는 현 지침 아래에서는 Windows/macOS 실측을 수행하지 않는다.

## 번외 실험

`experiments/binary_embedding/`와 `results/binary_embedding/`은 별도 실험이다.
고정된 바이트 특징에서 지도학습한 모델이 미사용 바이너리의 연산을 81.25%로
구분했다. 출력 표 예측은 76.65%였으며 숫자 입력 일반화는 검증하지 않았다.
이것을 바이너리 실행기나 Rust 동작을 배운 모델로 취급하지 않는다.
프로토콜·한계·원자료는 해당 디렉터리에 보존돼 있다.
