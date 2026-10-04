# Laya 결정 모델 시험

이 문서는 이전 시험 기록이다. 후속 사용자 요청으로 Laya 가중치를 제거하고
Gemma를 복원했다. 현재 기준은 [GEMMA_BASELINE.md](GEMMA_BASELINE.md)이며
아래 Laya 설치기는 Gemma를 제거하므로 현재 환경에서 실행하지 않는다.

## 후속 사용 적합성 판정

현재 multilingual 체크포인트는 **프로젝트의 자동 요청 처리 모델로 채택하지 않는다.**
기본 분류 추론은 정상 동작했지만 기능 선택과 인자 해석이 불안정했다.
이 판정은 현재 체크포인트/질문 형식/측정 범위에 대한 것으로, 모든 Laya 변형이나
미세조정 이후의 능력이 불가능하다는 증명은 아니다.

새 요청 24개를 추론 전에 고정하고 후보 순서 3개를 미리 정해 측정했다.
기능 선택에는 caller가 입력 타입을 제공해 후보가 3개(숫자: sub/xor/none,
파일: fill/map/none)로 줄어드는 유리한 조건을 사용했다. 별도 인자 진단은
정답 기능 계약을 미리 제공해 기능 선택 오류를 제외했다. 정답 인자 자체는
모델에 주지 않았다. 모든 기능 설명/역할은 프로그램에서 추출하고 원문 리터럴만
선택지로 사용했다. 작업별 Python 정답 로직이나 실행 절차는 추가하지 않았다.

| 진단 | 결과 |
|---|---|
| 한·영 결제/오류 기본 분류 control | 4/4 |
| 알려진 기능 선택 | 17/48회, 16개 요청의 순서 반복 |
| 모든 순서에서 정확한 알려진 기능 | 3/16개 요청 |
| 미지원 거절 | 13/24회, 8개 요청의 순서 반복 |
| 정답 계약을 제공한 숫자 인자 선택 | 27/64회 |
| 정답 계약을 제공한 경로 선택 | 6/16회 |
| 복수 후보가 있는 실행 폭 선택 | 3/16회 |
| 전체 인자 선택이 맞은 요청 | 0/16개 |
| 후보 순서에 따라 기능 답이 달라진 요청 | 6/24개 |

전체 폭 선택 11/24회 중 8회는 단일 선택지라 의미 해석 성능에서 제외했다.
180회 결정에서 입력 잘림 표시 0회. 요청 수와 반복 결정 수를 구분하며 이
작은 진단을 범용 정확도나 통계적인 성능 증명으로 취급하지 않는다.
미지원 AND/곱셈/OR를 XOR로 선택하는 오판도 있어 자율 실행을 승인할 수 없다.
control 4/4는 기본 추론이 작동한다는 확인이지 한국어 전반의 정확성 증명이 아니다.

후보를 제안하는 참고 역할에는 연결 가능하지만 독립 근거 검증이 필요하고,
실제로 도움이 되는지는 아직 검증하지 않았다. 대화/새 계약/학습 예제를 생성하는
기능은 현재 SDK의 선택·점수·확률 인터페이스에서 제공하지 않는다.
도메인 미세조정은 별도 과제이고 이번 검증에서는 수행하지 않았다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e USE_TF=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vectorpro-test python -m experiments.laya_suitability --output results/laya_suitability/reproduction
```

기존 결과 디렉터리를 덮어쓰지 않는다. 재현은 같은 요청의 replay이며 독립 첫
평가라고 표현하지 않는다. 원문/질문/확률: `results/laya_suitability/first-use`.
집계/SHA/한계: `results/laya_suitability/verification.json`.
관련 기존 회귀 32 passed, 13.60초. 핵심 학습기/커널/저장 형식 변경 없음.
기존 컨테이너와 설치된 체크포인트만 사용했고 실행·학습·등록을 하지 않았다.
판정 이후 모델/질문/데이터 튜닝 없음.

사용자가 Gemma 제거와 Laya 사용을 요청했다. 기존 `vectorpro-test` 하나에서
Laya multilingual만 설치하고 MiniLM과 배운 텐서 프로그램은 유지했다.
[공식 모델 카드](https://huggingface.co/convaiinnovations/laya)에 따르면 Laya는
문장을 생성하지 않고 주어진 질문의 선택지/점수/확률을 반환하는 결정 모델이다.
새 계약이나 학습 예제를 생성하는 ChatModel로 취급하지 않는다.

## 설치·공간

SDK `laya==0.3.26`, repo `convaiinnovations/laya`, revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`, subfolder `multilingual`.
`/opt/vectorpro-models/laya-multilingual`에 직접 저장해 Hub 캐시에 복제하지 않는다.
가중치와 토크나이저 포함 678,201,636바이트(약 678MB), Gemma GGUF와 manifest
제거 대비 391,105,004바이트 감소. 이는 논리 파일 크기이며 Windows의 가상
디스크가 자동으로 같은 만큼 축소됐다는 뜻은 아니다. C: 측정 여유 약 6.68GB.
파일 SHA-256과 설치 정보는 `results/laya_model/replacement.json`에 있다.

```powershell
nerdctl ps -a --filter name=vectorpro-test
nerdctl exec vectorpro-test python -m pip install --no-cache-dir laya==0.3.26
nerdctl exec vectorpro-test python experiments/setup_laya_model.py
```

설치기는 정확히 지정된 Gemma 두 파일만 제거한다. 컨테이너 삭제/이미지 생성은
없다. 가중치는 Git에 넣지 않는다. Qwen/Gemma 과거 결과는 재현 기록으로 보존한다.

## 실제 결정·메모리 실행 검증

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e USE_TF=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vectorpro-test python experiments/local_laya.py --output results/laya_model/reproduction.json
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_local_model_protocol.py tests/test_agent.py tests/test_semantic_catalog.py
```

출력 파일이 이미 있으면 덮어쓰지 않는다. CPU eager, 4 threads. SDK wheel에서
`backend="eager"`는 `laya.backends` 누락 오류가 났으므로 backend 인자를 생략하고
`fast=False, compile=False`를 사용했다. upstream 패치는 없다.

모델은 텐서에서 추출한 전체 기능 설명 중 하나와 폭을 선택하고, 선택된 기능의
입력 계약에 따라 원문 경로/숫자 중 인자를 선택한다. 파일명 숫자는 제외한다.
선택지 형식만 구성하며 특정 작업의 정답이나 인자 역할을 Python으로 결정하지
않는다. 실제 결정은 모두 Laya다. 기대 결과와 caller 예제는 모델에 제공하지 않는다.
이 실험은 AgentSession이 아니고, 모델 선택의 오류를 관찰하기 위한 별도 replay다.
근거 검증을 통한 운영 승인 대신 새 MemoryHostContext에서만 실행한다.
native 파일 쓰기·학습·등록·프로그램 저장은 하지 않는다.

기존 경로 6건과 바인딩 8건 중 실행 기대 11건은 **0/11** 성공했다. 기능 선택,
숫자 역할, 경로 선택 오류가 관찰됐다. 나머지 3건은 단순 needs_input 조건을
통과했으나 인자 부족으로 질문한 것이므로 올바른 미지원 의미 판단으로 간주하지
않는다. 22회 결정에서 usage의 잘림 표시는 0건이다. 점수에 맞춘 후속 튜닝은 없다.
독립 첫 평가도, 기존 Gemma의 근거 검증/후보 제한 루프와 공정한 직접 비교도 아니다.
이 결과로 Laya 전체의 성능을 단정하지 않지만 현재 연결을 운영 경로로 채택하지 않는다.
원문 질문/모든 선택 확률/메모리 결과는 `results/laya_model/replay.json`에 있다.
관련 기존 테스트 **32 passed, 11.06초**. 학습기·커널·저장 형식은 변경하지 않았다.

현재 진행 단계는 공통 기능 계약과 선택 어댑터 검증이다. 계약 통합, 독립적인
정답 근거, 입력 역할 검증, 자동 학습 준비가 남았다. macOS는 보류한다.
