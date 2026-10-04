# 계약 초안의 학습과 등록

초안 → 격리 학습 → 분리된 예제 검증 → 계약 조건 비교 → 등록 → ID 호출 순서다.
`runtime.teach_contract(draft, lesson=examples)`는 숫자 예제,
`runtime.teach_contract(draft, state_lesson=examples)`는 전체 파일 상태 예제를 받는다.
초안만 주면 `needs_learning_examples`이며 기능을 등록하지 않는다.

```json
{
  "version": 1, "name": "new_transfer",
  "description": "Copy source bytes to destination",
  "parameters": [
    {"name": "source", "type": "path", "role": "source file"},
    {"name": "destination", "type": "path", "role": "destination file"}
  ],
  "output": {"type": "value", "width": "W"},
  "allowed_operations": ["file.read", "file.write"]
}
```

초안은 위 여섯 필드만 받는다. 매개변수는 1~3개, 타입은 `value/path/buffer`,
출력 폭은 `W/W+1/2W/1`이다. 코드를 받지 않으며 기존 이름은 덮어쓰지 않는다.
상태 예제는 `experiments/requests/learn_transfer.json`의 `state_lesson` 형식이다.
디렉터리 상태는 `before_directories/after_directories` 상대 경로 목록으로 표현한다.
전체 파일 및 디렉터리 집합을 비교한다. [LINUX_TARGET.md](LINUX_TARGET.md)를 참고한다.
숫자 예제는 `training/validation` 각각에 `width/operands/targets`를 제공한다.
상태 예제의 `buffer` 입력은 hex이며 `list_loops`는 제한된 목록 반복 탐색을 활성화한다.
허용 호스트 작업은 후보 탐색에도 적용하고 채택 후 다시 확인한다.

학습은 복제 레지스트리와 메모리 호스트에서 진행한다. 예제를 통과해도 타입·폭이나
허용 호스트 작업 조건을 어기면 등록하지 않는다. 실패 시 실제 파일, 버퍼, 기존
레지스트리와 런타임 난수 상태는 그대로다. 성공 시 복제 레지스트리를 설치하므로
외부에서 보관한 이전 레지스트리 참조는 갱신되지 않는다. 동시 변경은 지원하지 않는다.

후보 수, 학습 횟수, 증거 크기와 최대 60초 한도가 있다. 숫자 학습의 시간 검사는
학습 후 수행하므로 정확히 60초에 강제 중단하지 않는다. 초과한 후보는 거절한다.
전역 PyTorch 난수 상태의 복구는 보장하지 않는다.

등록 결과의 `contract.id`를 `call_contract(id, {"source":"in", "destination":"out"}, 16)`에
사용한다. 호출에는 LLM이 필요하지 않다. 인자 이름·초안·증거 출처는 같은 프로그램
파일에 저장한다. 역할은 제출자의 선언이며 `role_source`에 표시한다. 기존 텐서에서
계산한 `argument_roles`를 선언으로 덮어쓰지 않는다. 기존 파일의 계약 호환성을 유지한다.

CLI JSON 요청은 `action: "teach_contract"`, `draft`와 하나의 예제를 받는다.
`registered`에만 저장하고 종료 코드 0, 예제 부족·학습 실패는 코드 2다.
LLM 도구도 같은 이름을 사용하며 출처를 항상 `llm_proposed_examples`로 기록한다.
등록 결과로 세션을 마치며 실제 파일 작업을 실행했다고 주장하지 않는다.
학습 비활성화 시 도구를 숨기고 직접 호출도 거절한다.

## 검증과 한계

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_contract_learning.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

전용 테스트 14개는 숫자 NAND 학습 뒤 미사용 16비트 100개 호출, 파일 복사 학습,
JSON/PT 재로드, 다른 호스트 루트 호출, 실패 시 무변경, 어댑터와 CLI 등록을 검증한다.
어댑터는 스크립트 모델로 검증했으며 실제 Gemma의 새 초안 작성 성공률은 측정하지 않았다.
최종 전체 회귀는 동일 컨테이너에서 223 passed, 194.44초였다.

`registered`는 제공된 예제와 구조 조건의 통과다. 선언이나 LLM 정답을 독립적으로
검증했다는 의미가 아니다. `intent_independently_verified`는 false다. 범용 정확성을
보장하지 않는다. 다음 단계는 Gemma 초안 작성 평가와 독립 목표 검증의 연결이다.
