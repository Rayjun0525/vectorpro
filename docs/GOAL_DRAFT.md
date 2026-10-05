# 자연어 목표 초안과 호출자 검토

2026-10-05. 작은 모델이 자연어 요청을 GOAL_EVIDENCE 형식으로 번역한다.
이 번역 결과는 승인된 목표가 아닌 needs_goal_review 상태의 초안이다.
모델이 스스로 승인하거나 관찰·학습·실행을 시작하는 도구는 제공하지 않는다.

```python
from vectorpro.goal_draft import propose_goal, accept_goal
draft = propose_goal(model, providers, intent)
# 호출자는 원문과 draft['review']의 모든 조건을 비교한다.
# 동의한 경우에만 정확한 proposal_sha256을 전달한다.
goal = accept_goal(draft, caller_approved_sha256)
session = AgentSession(runtime, model, program_path,
    reference_providers=providers, request_goal=goal)
```

확인은 신뢰하는 호출자 코드에서 이뤄져야 한다. accept_goal을 모델 도구로 노출하거나
초안 해시를 자동 복사하면 사람의 확인을 대체하지 못한다. 해시는 검토된 내용의
동일성을 검사하며 신원 인증이나 자연어 의미의 증명이 아니다. 기존 API로 직접
goal을 제공하는 것도 여전히 호출자의 책임이다. 같은 모델을 한 번 더 불러 찬성하는
결과만으로 독립 검증했다고 표현하지 않는다.

초안은 요청 원문을 백엔드가 넣고 모델은 조건만 제출한다. 규칙은 기존의 부재·존재·
보존·초기 원본과 바이트 일치·지정 항목 외 전체 상태 보존을 사용한다. 파라미터는
설치된 기준들이 공통으로 사용하는 path 인터페이스에서만 선택할 수 있다.
서로 다른 인터페이스는 needs_input으로 반환한다. 관찰 기준의 설명이나 실행 argv를
초안 문맥에 넣지 않으며, 기준을 고르기 전에 목표를 작성한다.
상태 조건과 정확히 한 개의 전체 보존 조건이 필요하다. 숨은 파일·폴더 변경을
단순히 무시하는 목표가 되지 않도록 검토에 허용된 변경 항목 전체를 표시한다.

한 번의 모델 호출만 허용한다. 지원하지 않는 암호화/바이트 변환/프로세스/네트워크/
숫자 목표와 모호한 요청에는 ask_user를 사용하도록 안내한다. 모델이 이 안내를
잘 따르는지는 실험으로 측정하며 성공을 보장하지 않는다. 문법적 유효성만으로
목표 의미가 옳다고 승인하지 않는다. 잘못된 도구, 여러 호출, 없는 인자 이름,
부족한 조건은 goal_draft_failed가 된다. 조건 순서도 해시에 포함된다.

CLI: 기존 LLM 어댑터에 --draft-goal과 --reference-providers FILE을 제공하면
검토 가능한 JSON을 출력하고 종료 코드2로 끝난다. --request-goal, --encoder,
--acquisition-evidence와 함께 사용하지 않는다. 프로그램 저장이나 사용자 파일
작업은 하지 않는다. 확인된 goal은 다음 실행의 --request-goal FILE로 전달한다.
draft JSON 전체를 goal로 전달하면 형식 검사에서 거절한다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_goal_draft.py tests/test_goal_evidence.py tests/test_verified_reuse.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_draft --output results/goal_draft_gemma_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

실험은 모델 추론 전 cases.json에 원문과 기대 조건을 고정한다. 기대 조건은 모델에
노출하지 않는다. 상태 조건의 순서와 변경 예외 목록의 순서는 평가 시 정규화한다.
실제 Gemma는 초안만 평가하며 승인·관찰·학습·실행하지 않는다. 테스트에서는
스크립트 모델과 호출자 확인으로 이동 계약 실행 및 저장 후 LLM 학습 없이 반복
실행하는 연결 경로를 확인한다. 실제 인간의 확인이 이루어진 실험은 아니다.

현재 위치는 자연어 목표를 검토 가능한 형태로 만드는 단계다. 사람이 제공하는
판단 기준을 완전히 제거한 것은 아니다. 다음은 더 넓고 독립된 요청 평가,
인자 역할/실제 입력의 독립 확인, 프로세스·네트워크 목표 표현 확대다.
사용 횟수 자체가 모델 가중치를 갱신하지 않는다. 확인되고 검사된 목표와 계약을
축적해 반복 요청의 재해석·재학습 비용을 줄이는 방향이다.

실제 Gemma 첫 초안 평가는 1/3였다. 복사는 잘못된 조건과 전체 보존 조건 누락으로
거절됐고, 이름 변경도 다시 질문했다. 미지원 암호화의 질문 전환만 평가를 통과했다.
실행이 차단된 것을 목표 이해 성공으로 세지 않는다. 결과는 HANDOFF.md에 보존한다.
