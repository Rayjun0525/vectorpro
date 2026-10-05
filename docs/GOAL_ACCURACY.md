# Gemma 목표 초안 정확도 개선 기록

2026-10-05. 기존 Gemma3 1B Q8_0와 vectorpro-test 하나를 그대로 사용했다.
모델 다운로드/이미지 빌드/학습은 하지 않았다. 이번 변경은 입력 문맥과 출력 형식의
개선이며 가중치 학습이나 텐서 실행 능력의 확장이 아니다.

## 오류와 변경 방법

기존 rules 방식은 모델이 조건 종류·인자 역할·원본 바이트·전체 보존 조건을
동시에 조립해야 했다. 첫 평가 results/goal_draft_gemma는 1/3이었다.
복사는 source에 absent와 present를 동시에 넣고 보존 조건을 누락했다.
이름 변경은 파일명이 명시됐는데도 질문했으며 암호화는 질문으로 전환했다.

1. 실험 출력을 인자별 states로 바꿨다. 각 인터페이스 인자는 unchanged/absent/present/
   initial:다른인자 중 하나를 선택한다. 모든 인자를 빠짐없이 제공해야 한다.
   백엔드는 이를 기존 goal 규칙으로 변환하고 unchanged인 항목을 제외한 변경 목록으로
   전체 보존 조건을 구성한다. 문자열 작업 이름을 보고 절차를 구현하지 않는다.
   역방향 원본 선택, 삭제, 전체 보존도 같은 변환으로 표현한다.
2. 상태 표기의 짧은 예시를 추가했다. 인자 이름은 설치된 인터페이스에서 가져오며
   평가 파일명이나 기대 정답은 넣지 않는다. 예시는 기존 바이트 보존/원본 부재/
   다른 인자로 초기 바이트 전달/미지원 목표의 표기를 설명한다. 이는 프롬프트 예시이며
   텐서 학습 또는 독립 검증이라고 부르지 않는다.
3. 지원 범위 판단과 상태 작성의 문맥을 분리했다. 첫 호출은 요청 전체를
   supported/unsupported/ambiguous로 분류한다. 뒤 두 상태는 질문으로 종료하고,
   supported만 두 번째 호출로 인자별 최종 상태를 작성한다. 같은 모델의 판단이므로
   독립 인증이 아니다. 기존 rules 방식은 encoding="rules"로 한 번의 호출을 유지한다.
4. 전체 보존만 요청할 수 있도록 unchanged_except의 빈 목록을 허용했다.
   빈 목록은 모든 파일과 빈 디렉터리가 보존돼야 함을 뜻한다. 기존 저장 파일을
   수정하지 않으며 goal/검토/해시/호출자 확인 형식은 그대로 유지한다.

states도 잘못된 의미를 출력할 수 있다. 문법 제한은 의미 정확성을 보장하지 않는다.
propose_goal/assess_goal에는 승인·학습·실행 기능이 없다. accept_goal의 호출자 확인,
관찰 목표 검사와 기존 계약 검증은 계속 필요하다. 모델 판단 결과를 자동 승인하면
이번 개선으로 독립 검증이 생기지 않는다.

## 평가 설계와 시행착오

기존 요청 재시험은 실험을 조정한 뒤 같은 문장을 다시 본 것으로 독립 정확도가 아니다.
초기 결과와 모든 중간 raw_calls/summary를 삭제하지 않았다.

| 방식 | 기존 3건 재시험 | 결과 경로 |
|---|---:|---|
| 최초 규칙 목록 | 1/3 | goal_draft_gemma |
| 인자별 상태만 단순화 | 1/3 | goal_draft_states_replay |
| 인자별 상태 + 표기 예시 | 1/3 | goal_draft_states_examples_replay |
| 지원 판단 분리 + 표기 예시 | 2/3 | goal_draft_phased_replay |
| 인자별 판정 분리 | 1/3 | goal_draft_per_parameter_replay |
| 객관식 문자 + 예시 | 1/3 | goal_draft_choice_replay |
| 짧은 객관식 문맥 | 0/3 | goal_draft_compact_choice_replay |
| 의미가 드러나는 선택 표기 | 0/3 | goal_draft_word_choice_replay |

수치는 기대 조건과 정규화된 JSON 조건의 일치 및 미지원 목표의 질문 전환 점수다.
조건 순서와 예외 목록 순서는 무시한다. 논리적 동등성 전체를 판정하지 않는다.
예시만 추가한 복사 초안은 source를 initial:source로 표현해 동일 바이트를 요구했지만
기대 unchanged 조건과 달라 이 점수에서는 실패했다. 따라서 모든 점수 실패가
의미적 오류를 뜻하지 않는다. 반대로 암호화의 일반 복사·이동 조건 대체는 실제 오류다.

최종 재시험은 복사/이름 변경을 맞췄지만 암호화를 supported로 판정해 잘못된 초안을
냈다. 최초 방식의 암호화 질문 전환보다 나빠진 부분이다. 2/3만 제시해 안전성까지
개선됐다고 주장하지 않는다. 실제 승인/사용자 파일 실행은 하지 않았다.

추가 8건은 experiments/goal_draft_heldout.json에 기대 조건까지 고정해 두고
기존/최종 방식에 동일하게 제공했다. 영어/한국어 복사와 이동, 전체 보존, 대상 삭제,
바이트 순서 변환과 모호한 요청을 포함한다. 기대 조건은 모델에 제공하지 않는다.
states 방식은 이 평가를 본 뒤 다시 조정하지 않았다. 규칙 방식과 states 방식 모두
2/8이었다. 인자별/객관식의 추가 실험은 별도 재시험으로 남겼다. 제한된 소표본이며 더 넓은 범용 정확도
보장이나 첫 리눅스 목표 전체의 완료율이 아니다. 결과와 최종 회귀는 HANDOFF.md에 기록한다.

## 채택한 개선: 확인된 목표 예제의 의미 벡터 검색

Gemma 자체의 목표 작성 정확도가 안정적으로 오르지 않아, 선택한 개선 경로는
GoalMemory의 예제 검색이다. 고정 예제13개를 experiments/goal_memory_seed.json에
작성했다. 호출자가 제공한다는 형식의 교사 데이터이며 이번 실험에서는 우리가 작성한
라벨이다. 실제 사용자 확인을 거쳐 축적한 경험이라고 주장하지 않는다. 복사/이동/
두 파일 보존/대상 삭제와 미지원·모호한 목표의 영어·한국어 예제를 포함한다.
예제는 절차 코드가 아니라 요청과 최종 상태 조건이다.

이미 설치된 multilingual MiniLM-L12-v2로 문장 벡터를 얻는다. 가중치를 학습하거나
새 모델을 내려받지 않는다. 목표 조건이 같은 예제들을 그룹화하고 그룹별 최고
코사인 유사도를 비교한다. 최소유사도0.55, 다른 목표 그룹과 차이0.03을 만족해야
초안을 제안한다. 이 수치는 최초 예제 검색 실험 전에 정했으며 이후 평가를 보고
바꾸지 않았다. 확률·정확도 보장이 아니다. 불충분한 유사도, 다른 목표와 경합,
미지원 예제의 일치는 질문으로 종료한다. 요청 원문은 그대로 초안에 들어간다.

예제·벡터·인코더 식별 정보·한도는 같은 런타임 프로그램 파일의 goal_memory로
저장한다. .pt는 기존 tensor_codec을 통해 모두 텐서로 저장되고 .json도 호환한다.
재로드 시 예제를 다시 임베딩하지 않으며 조회 문장만 인코딩한다. 인코더 식별 정보가
다르면 거절한다. 목표 검색 데이터는 승인된 요청 연결이나 실행 계약을 변경하지 않는다.
반환은 항상 needs_goal_review 또는 needs_input이다. accept_goal 이후 기존 기준
관찰·목표 검사·재사용 검사를 거쳐야 한다. 유사도를 승인 또는 의미 증명으로 쓰지 않는다.

API는 propose_goal(..., goal_memory=memory) 또는 memory.propose(intent)다.
GoalMemory.attach(runtime)로 같은 파일에 저장하고 from_runtime으로 복구한다.
LLM의 기존 기본 rules 방식은 유지했다. states/per_parameter는 비교 실험용 선택이다.
CLI --draft-goal --goal-memory-encoder LOCAL_DIRECTORY는 저장된 예제를 읽어
LLM 호출 없이 초안을 내며 코드2로 종료한다. 이 모드에서는 endpoint/model이 필요 없다.

처음8건의 진단은 4/8(results/goal_memory_probe.json). 이를 본 이후 예제/임계값을
바꾸지 않고 experiments/goal_accuracy_validation.json의 새로운8건을 고정했다.
이 새 평가에서 Gemma 규칙 방식은2/8, 예제 검색은6/8이었다. 같은 문장을 같은
조건으로 평가한 비교다. 새로운 평가 문장은 특정 파일명 대신 원본/대상 역할 표현을
주로 사용하며, 최초8건과 난도가 같다고 가정하지 않는다.

지원 목표6건에서는 Gemma0/6, 검색4/6이 정확했다. 검색은 이동1건을 잘못 제안하고
두 파일 보존1건은 질문으로 돌렸다. 미지원·모호한2건은 두 방식 모두 질문으로 돌렸다.
Gemma는 잘못된 초안0건이지만 지원 목표를 하나도 맞추지 못했고, 검색은 제안5건
중4건이 맞았다. 75%를 곧바로 실행 가능한 신뢰도로 해석하면 안 된다. 소표본이며
학습 목표의 범위와 파일명/역할/부정 표현이 달라지면 정확도도 달라질 수 있다.

가중치 학습이 아닌 확인된 예제의 축적으로 재사용 범위를 늘리는 방법이다.
잘못된 라벨을 추가하면 오류도 축적된다. 자동으로 모델 출력에 라벨을 붙여 다시
검색에 넣지 않는다. 실제로 경험을 쌓을 때에는 검토·검증된 목표만 추가해야 한다.

results/goal_memory_execution은 기존 진단 요청의 이동 초안을 고정된 기대 조건으로
확인하는 스크립트 호출자를 사용했다. 새로운 파일 바이트로 실제 이동하고, 저장 후
같은 요청을 추가 관찰·학습 없이 단일 계약 호출하는2/2 경로를 확인했다.
이 경로 검증은 실제 인간 검토나 새로운 자연어 정확도 측정이 아니다.

## 재현과 비용

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_draft --encoding rules --suite heldout --output results/goal_draft_rules_heldout_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_draft --suite heldout --output results/goal_draft_phased_heldout_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_goal_states.py tests/test_goal_draft.py tests/test_goal_evidence.py tests/test_verified_reuse.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_draft --encoding rules --cases-file experiments/goal_accuracy_validation.json --output results/goal_rules_validation_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_draft --goal-memory --cases-file experiments/goal_accuracy_validation.json --output results/goal_memory_validation_recheck
```

결과 경로는 기존 기록을 덮어쓰지 않는 새 이름을 사용한다.
raw_calls.json에 모델 호출의 usage/seconds/전체 입력과 출력을 저장했다.
states는 보통 최대2회, per_parameter는 인자 수만큼, rules는1회 호출한다.
GoalMemory는 LLM을 호출하지 않으며 조회 문장 인코딩과 벡터 내적을 수행한다.
이미 확인·등록한 동일 요청은 기존 계약을 재사용한다. 모델 평가와 회귀를 함께
실행한 경우의 경과 시간은 전용 환경의 속도 비교로 해석하지 않는다.

남은 과제는 미지원 요청의 오해, 독립된 새 요청 평가 확대, 여러 인터페이스와
입력 역할의 확인, 프로세스·네트워크 목표 표현 확장이다.
