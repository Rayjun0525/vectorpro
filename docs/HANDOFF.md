# 작업 인수인계

기록 기준: 2026-10-04, 한국 시간. Git의 `main` 작업 트리에 구현과 실험 결과가
미커밋 상태로 있다. 이 문서 작성 시 새 커밋이나 PR은 만들지 않았다.
변경된 소스뿐 아니라 새 파일과 결과 자료도 함께 검토해야 한다.

## 현재 구현

| 위치 | 역할 |
|---|---|
| `src/vectorpro/runtime.py` | 알려진 요청 실행, 미지 요청 학습, 단일 파일 저장·재로딩 |
| `src/vectorpro/__main__.py` | JSON 요청 CLI. `python -m vectorpro` 또는 설치된 `vectorpro` |
| `src/vectorpro/host.py` | 버퍼·파일 실행 기능, 격리된 메모리 파일 환경, 기본 타입 규약 |
| `src/vectorpro/learning/examples.py` | 정답 함수 없는 숫자 예제 `ExampleLesson`과 검증 |
| `src/vectorpro/learning/learner.py` | 기존 규칙·조합·bit-fold 학습과 유한 예제 학습 |
| `src/vectorpro/learning/stateful.py` | 초기/목표 파일 상태로 타입 기반 순차 절차 탐색 |
| `src/vectorpro/machine/` | 호출·분기·반복 실행. `calls` 텐서로 반환값 없는 호출 지원 |
| `src/vectorpro/learning/registry.py` | 기능 보관·설명·주소 해석·저장, 효과 있는 기능의 계산 탐색 제외 |

상태 학습은 입력 타입, 초기 파일, 정확한 최종 파일, 선택적 숫자 반환값을 받는다.
사용자는 호출 이름·순서·인자 연결을 제공하지 않는다. 후보를 가상 환경에서
실행하고 사례를 만족하는 순서를 일반 `VectorProgram`으로 저장한다.
기본 OS 동작과 타입 카탈로그는 실행 기반으로 사람이 제공한다.

## 최근 검증

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

## 재현

컨테이너 규칙은 루트 `AGENTS.md`를 따른다. Python, PyTorch, pytest, Rust,
gcc, binutils는 같은 컨테이너에 설치돼 있다. `/workspace`는 이 저장소다.

```powershell
nerdctl ps -a --filter name=vectorpro-test
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/stateful_learning.py
```

`experiments/stateful_learning.py`는 소유한 `results/stateful_model/` 데모 자료를
재생성한다. `summary.json`에 측정값과 발견한 절차가 있으며 `program.json`은
그 기능을 담은 단일 프로그램 파일이다. 이 자료를 사용자 운영 파일로 쓰지 않는다.

숫자 학습 CLI 예제는 `experiments/requests/learn_xor.json` 및 `run_xor.json`,
상태 학습 요청 예제는 `learn_transfer.json`이다. 새 숫자 학습을 재현할 때는
아직 해당 기능이 없는 프로그램 파일을 사용한다. 이미 있으면 학습을 건너뛴다.

## 한계와 다음 범위

- 상태 탐색은 제한된 길이의 순차 호출만 지원한다. 상태 기반 분기·가변 길이
  반복·복잡한 자료구조의 학습은 남아 있다.
- 산술은 기존 범용 구조 후보와 bit-fold 검색을 사용한다.
- LLM 도구 연결, 자연어 해석, CPU 실행 기록에서의 동작 학습은 미구현이다.
- Linux 컨테이너에서 검증했다. Windows/macOS 설치·동일 동작 검증은 남아 있다.
- 우선 초기모델을 확실히 한다. 초기모델 이후의 운영 기능을 성급히 확장하지 않는다.
- `ls`는 사용자가 가능성을 질문한 사례이지 현재 구현 과제가 아니다.

## 번외 실험

`experiments/binary_embedding/`와 `results/binary_embedding/`은 별도 실험이다.
고정된 바이트 특징에서 지도학습한 모델이 미사용 바이너리의 연산을 81.25%로
구분했다. 출력 표 예측은 76.65%였으며 숫자 입력 일반화는 검증하지 않았다.
이것을 바이너리 실행기나 Rust 동작을 배운 모델로 취급하지 않는다.
프로토콜·한계·원자료는 해당 디렉터리에 보존돼 있다.
