# 호출자 목표와 기준 관찰의 일치 검사

2026-10-05. 자연어 요청을 독립적으로 이해하는 기능은 아니다. 호출자가 제공한
구조화된 목표를 모델의 기준 선택과 별도로 검사한다. 이 기능에는 새 모델이 필요 없다.
LLM이 생성한 목표를 그대로 제공하면 그 목표 해석의 오류는 여전히 남는다.

```json
{"intent":"Rename source.bin to target.bin, removing source.bin.","rules":[
  {"kind":"equals_initial","parameter":"destination","source":"source"},
  {"kind":"absent","parameter":"source"},
  {"kind":"unchanged_except","parameters":["source","destination"]}
]}
```

intent는 원문과 정확히 같아야 한다. parameter/source는 설치된 기준 인터페이스의
인자 이름이다. 실제 파일 경로는 기준 사례의 가상 입력에서 치환한다. 규칙은
실행 절차가 아니라 최종 상태의 조건이다. Python 작업 순서를 제공하거나 커널에
복사/이동 전용 분기를 넣지 않는다.

지원 조건은 absent(파일·폴더 부재), present(파일·폴더 존재), unchanged(해당 항목
존재와 내용·종류 보존), equals_initial(대상 파일 바이트가 초기 원본 파일과 같음),
unchanged_except(지정 항목을 제외한 전체 파일과 디렉터리 보존)이다.
조건 1~32개, 인자 목록 최대16개다. 명시한 조건만 검사하므로 불완전한 목표가
사용자 의도를 완전히 증명하지 않는다. 시간/프로세스/네트워크/숫자 반환의 목표 검사는
이번 형식에 없다. 기존 메모리 실행 평가와 전체 스냅샷 비교는 계속 적용된다.

ReferenceProviders가 관찰한 학습·검증·숨긴 사례 모두를 조건과 비교한다.
하나라도 모순이면 근거 bank를 만들지 않으며 학습/재사용/저장/실제 실행으로 가지 않는다.
한 번 실패한 기준을 모델이 다시 선택하도록 허용하지 않는다. 모델은 질문하거나
미실행 상태로 끝낸다. 실제 사용자 파일은 이 관찰과 목표 검사에 사용하지 않는다.

재사용 intent_bindings와 새 학습 acceptance에는 goal_sha256을 추가한다.
현재 목표의 해시가 다른 연결 또는 목표 검증 없이 저장된 기존 연결은 목표 검사 모드에서
승인하지 않는다. 기존 v1 파일은 계속 읽으며 목표 없는 기존 동작은 호환성을 유지한다.
같은 목표로 승인된 동일 원문 요청은 재로드 후 수집·학습 없이 계약을 실행할 수 있다.
이 검사는 유한한 관찰 사례와 현재 목표 해시의 기록이지 범용 의미론 증명이나 인증 서명이 아니다.

사용 방법:

```python
bank = EvidenceBank(records, goal=caller_goal)
session = AgentSession(runtime, model, program_path,
    reference_providers=providers, request_goal=caller_goal)
```

직접 bank를 제공할 때에는 bank에 goal을 넣는다. 두 경로로 중복 제공하면 거절한다.
CLI에는 --request-goal JSON_FILE을 추가했다. --acquisition-evidence 또는
--reference-providers와 사용한다. 카탈로그 모드와 독립 근거 모드는 기존처럼 구분한다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_goal_evidence.py tests/test_verified_reuse.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.verified_reuse --caller-goals --output results/goal_evidence_protocol_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.verified_reuse --caller-goals --live-gemma --output results/goal_evidence_gemma_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

실험 목표는 모델 추론 전에 caller_goals.json으로 고정하며 이전 실험의 정상
reference_acquisition_gemma/program.pt에서 시작한다. 이전 실패한 재사용 파일은 쓰지 않는다.
passed는 요청 달성, rejected_cleanly는 실제 파일 변경·계약 호출 없는 거절이다.
거절을 목표 달성 성공으로 계산하지 않는다. 결과와 최종 회귀는 HANDOFF.md에 기록한다.

현재 실제 Gemma는 복사/반복 2/3 목표를 달성하고, 이름 변경에 선택한 복사를
검출해 파일과 프로그램을 변경하지 않고 거절했다. 저장된 목표 검증 연결은 복사
1개뿐이다. 이름 변경의 의미 선택 오류는 고쳐졌다고 주장하지 않는다.

사용 횟수 자체는 모델 가중치나 정확도를 높이지 않는다. 검증된 연결 축적으로
알려진 요청의 재사용 범위가 늘어난다. 잘못된 목표를 받아들이면 오류도 축적된다.
다음 단계는 자연어에서 목표를 만들고 사용자 의도와 일치하는지 별도 확인하는 경로,
프로세스/네트워크 목표 검사와 독립된 미사용 요청 평가다.
