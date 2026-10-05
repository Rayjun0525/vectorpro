# 후보의 차이를 확인하는 질문 시험

2026-10-05. 사용자의 ‘창의적 정확도 개선 방법을 시험하자’ 요청에 대한 실험이다.
이번 코드는 experiments에만 두며 실행 커널, 학습기, 저장 형식, 기본 GoalMemory의
선택 정책을 변경하지 않았다. 자연어 판단을 자동 승인하거나 사용자 파일을 실행하지 않는다.

## 방법과 경계

기존43개 예제/설치된 MiniLM으로 여섯 종류의 파일 목표 후보를 검색한다.
후보의 각 인자 최종 상태와 허용 변경 목록을 비교하고, 후보들을 가장 잘 나누는
조건 하나를 질문으로 만든다. 예를 들어 source가 보존/삭제되는지, destination에
어느 파일의 초기 내용이 들어가는지, 어떤 항목만 변경할 수 있는지를 확인한다.
이 조건은 목표 규칙에서 생성하며 Python 작업 절차를 새로 구현하지 않는다.

답변은 제시된 값 또는 outside/unclear 중 하나다. 답변으로 후보를 좁히고, 하나만
남아도 needs_goal_review이며 승인·학습·실행은 하지 않는다. 질문의 요청/후보/선택지와
SHA-256을 다시 대조해 바뀐 질문에 대한 답은 거부한다. 해시는 신뢰된 호출자 인증이
아니며, 실험용 프로토콜에서 오래된 질문/선택지가 섞이는 것을 검출하는 장치다.
질문에는 기대 정답을 넣지 않는다.

두 종류의 답변을 반드시 분리했다.

- **모의 호출자:** 평가 정답에서 해당 조건의 값만 답한다. 정답을 아는 사람이 추가
  정보를 제공할 때의 가능성 측정이다. 실제 사용자 답변이나 자율 정확도가 아니다.
- **Gemma:** 기존 Gemma3 1B Q8_0가 원문/조건/선택지만 보고 답한다. 기대 목표는
  모델에 제공하지 않으며 학습·승인·실행 도구도 주지 않는다.

기존 보류7건은 진단 재시험이다. 새10건(지원6/미지원·모순·모호4)은 추론 전에
cases.json에 고정했다. 언어와 역할 표현을 포함하고 같은 후보·질문 생성으로 비교했다.
임계값/교사43개는 변경하지 않았다. 양쪽 모두 마지막 목표 규칙을 정규화해 비교한다.
지원 요청의 보류는 실패이며 미지원 요청의 보류만 맞음으로 계산한다.

## 결과

최근접 목표 상위3개를 후보로 쓰는 첫 시험:

| 방식 | 기존 보류7건 맞음 | 새10건 맞음 | 새 지원 목표 맞음 | 새 오답 초안 |
|---|---:|---:|---:|---:|
| 기존 합의 분류, 추가 답변 없음 | 0/7 | 5/10 | 2/6 | 2 |
| 모의 호출자, 한 질문 | 6/7 | 9/10 | 5/6 | 1 |
| 모의 호출자, 최대 두 질문 | 7/7 | 9/10 | 5/6 | 1 |
| Gemma, 최대 두 질문 | 2/7 | 3/10 | 1/6 | 3 |

Gemma는 기존7건에도 오답 초안2건을 냈다. 실제17번 호출, 추론 시간 합160.52초,
usage total_tokens 합11,024. MiniLM 로딩/사례 인코딩 시간은 이 합에 포함하지 않는다.
전용 환경의 속도 비교나 실제 LLM 에이전트 전체 비용으로 해석하지 않는다.
모든 메시지/도구/응답/시간/토큰은 results/goal_questions/raw_calls.json에 보존했다.

새10건에서 기존 합의 방식도 오답2건을 냈다. 지난18건의 오답0건이 범용 보장이
아님을 다시 확인했다. 같은 Gemma에 질문을 쪼개주는 것만으로 자동 정확도가
개선된다는 근거는 없으므로 이 방식을 기본 경로에 넣지 않았다.

모의 호출자의 새 오답은 ‘두 파일을 모두 보존’하는 목표가 상위3개에서 누락된
경우다. 올바른 조건 한 개가 다른 목표와 공통이면, 답변이 맞아도 잘못된 후보가
혼자 남을 수 있다. ‘질문을 했으니 정확하다’고 취급하면 안 된다.

이를 본 뒤 전체 목표6개를 후보로 쓰는 **진단 대조 재시험**을 했다. 같은 사례를
재사용했으므로 독립된 새 정확도 평가에 합산하지 않는다. 모의 호출자가 최대 두
조건을 답하면 기존7/7, 새10/10이며 오답0건이었다. 한 질문만 쓰면 각각3/7,
6/10이므로 질문 수와 후보 누락 위험의 절충이 있다. 이 전체 후보 시험에는
Gemma를 다시 호출하지 않았다. 제한된 여섯 목표와 정확한 외부 답변이 있다는
조건에서의 결과이며, 임의 요청이나 사람 없이 작동하는 능력의 증거가 아니다.

집계: results/goal_question_report.json. 원본: results/goal_questions,
results/goal_questions_all_candidates. 최초34개 관련 테스트 중 변조 검사1개가
실패한 이유는 시험에서 선택지를 원래 값과 같은 값으로 ‘변조’했기 때문이다.
실제로 다른 값으로 바꾸도록 시험을 고친 뒤 **34 passed (3.58초)**를 확인했다.
질문을 JSON으로 주고받을 때도 같은 답변을 연결하도록 직렬화 표현을 정리했고
관련34개를 다시 확인했다(3.62초). 전체 회귀는 **392 passed (261.70초)**.
마지막 JSON 표현 정리 후에는 관련34개를 재검사했으며 학습기/커널/저장 형식은
변경하지 않았다.

## 가상 환경 실행 검증

별도 experiments/goal_candidate_preview.py는 기존 저장된 복사/이동 텐서 계약을
정방향/역방향 인자 연결로 호출했다. 새 MemoryHostContext를 매번 만들고 빈 바이트,
바이너리, 미사용 문자열의 세 입력을 사용했다. 총4개 경로/12회 실제 텐서 실행의
전체 파일/빈 폴더 결과를 기존 목표 검사기와 비교했다. 각각 정확히 한 목표와
일치했고 원본 프로그램 파일은 바뀌지 않았다. 학습·native 파일 접근은 없었다.
보존/삭제 전용 계약이 이 파일에 없어 여섯 목표 전체를 실행한 시험은 아니다.

가상 실행은 후보가 만드는 결과 차이를 보여준다. 어느 결과를 사람이 원하는지
판정하는 증거는 아니다. 목표 질문 결과와 실행 결과를 자동으로 연결하지 않았다.
기록: results/goal_candidate_previews/summary.json.

## 재현과 다음 단계

기존 vectorpro-test 하나/기존 이미지/설치 모델을 재사용했다. 결과 폴더는 새 이름을
사용한다. 이미지 빌드·컨테이너 생성·모델 다운로드는 없다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_question_benchmark --live-gemma --output results/goal_questions_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_question_benchmark --candidate-count 6 --output results/goal_questions_all_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_candidate_preview --output results/goal_candidate_previews_recheck
nerdctl exec vectorpro-test python -m experiments.goal_question_report
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_goal_questions.py tests/test_goal_memory.py tests/test_goal_evidence.py
```

집계 스크립트는 최초 기록 경로를 읽는다. 다른 경로로 재시험한 결과는 각 폴더의
summary.json에서 확인한다.

현재는 확인 질문과 후보 검증의 개념 실증까지 왔다. 다음은 실제 호출자가 조건에
답하는 API, 후보 집합 밖의 목표를 놓치지 않는 장치, 전체 목표의 명시 확인,
확인된 계약의 직접 재사용을 연결하는 것이다. 모델만으로 무인 해석하는 경우의
개선 방법과 더 넓은 리눅스 작업의 목표 표현은 계속 연구·구현해야 한다.
