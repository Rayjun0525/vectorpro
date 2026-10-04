# 순차 프로세스 연결과 실패 결과 분기

이 단계에서는 이미 배운 기능을 조합해 두 프로세스의 stdout/stdin 연결을
예제에서 발견한다. 중간 출력을 완전히 버퍼에 받은 뒤 다음 프로세스를 실행한다.
동시 스트리밍 OS 파이프는 아직 구현하지 않았다.

## 습득한 기능

| 기능 | 입력 → 출력 | 발견한 호출 구조 | 후보 |
|---|---|---|---:|
| run_stdout | 요청·stdin → stdout bytes | process.run → process.stdout | 30 |
| chain_stdout | 첫 요청·둘째 요청·stdin → 최종 stdout | run_stdout → run_stdout | 553 |
| save_stdout | 프로세스 결과·경로 → 쓴 길이 | process.stdout → file.write | 387 |
| save_success | 프로세스 결과·경로 → 성공 길이 또는 0 | process.code → 조건부 save_stdout | 1293 |

첫 기능의 결과를 둘째 호출의 입력으로 전달하는 순서와 인자 경로도 학습한다.
호출 횟수·허용 작업·타입과 최종 상태/출력 근거를 제공하며 작업 명령열은 제공하지
않는다. chain_stdout은 2개의 습득 기능 호출이고 실제 호스트 호출은 4개다.
save_success는 2호출과 제어 1단계이며, 성공 시 내부 stdout 추출과 쓰기를 실행한다.
모든 작업 로직은 저장된 호출·쓰기·제어 텐서에 둔다.

각 기능은 학습 2건과 독립 검증 2건을 사용한다. 입력/중간/최종 출력이 다른
tr 관찰과 엄격한 프로세스 요청·입력·순서 재생으로 우회/역순 후보를 구분했다.
이것은 외부 tr의 알고리즘을 학습하거나 바이너리를 대체한 것이 아니다.
Popen을 실패 함수로 바꾼 테스트에서 학습 중 프로세스를 띄우지 않음을 확인했다.

## 분기 탐색 범위

`control_flow: true, branches_only: true`는 일반 제어 문법에서 guard만 탐색한다.
조건 레지스터·극성·구간은 예제로부터 선택하며 특정 작업을 커널에 추가하지 않는다.
기본 branches_only는 false로 기존 branch/while 후보를 유지한다. 이미 습득한 기능
내부의 반복이나 별도의 목록/버퍼 반복 문법을 금지하는 설정은 아니다.

범용 branch+while 탐색으로 직접 3호출과 조합 2호출을 시도했으나 각각
1223·1257후보에서 시간 한도에 걸렸다. 학습된 save_stdout과 branches_only를
사용한 최종 탐색은 1293후보에서 성공했다. 후보 20000개/시간 60초 한도는 유지했다.
예비 실패 근거와 커리큘럼을 삭제하지 않았다.

새 분기 기능의 설명은 극성을 단정하지 않고 저장된 conditional routing으로
표현한다. 설명 버전 표식은 새 branches_only 기능에만 저장하며 기존 프로그램의
설명과 계약 ID는 유지한다. 실행 커널이나 저장/계약 버전은 바꾸지 않았다.

## native 검증과 실패

기존 process_results 프로그램에 네 기능을 축적해 저장·재로딩했다.
미사용 조합 cat→cat, cat→tr, tr→cat과 빈/바이너리/한글/6144바이트에서
전체 파일·디렉터리 상태와 결과 **4/4** 통과했다. 첫 명령/둘째 명령의 실행 파일이
없으면 그 단계에서 오류를 반환하고 이후 호출은 실행하지 않는 것도 검사했다.

별도 실제 프로세스의 성공(빈 출력·한글), 종료 19, 신호 종료 -15의 결과를
저장 분기에 전달해 **4/4** 통과했다. 성공 빈 출력은 파일을 만들고, 실패 출력은
기존 파일을 보존하며 쓰기를 실행하지 않는다. stderr와 raw exit code는 원본
프로세스 결과에 남는다. chain_stdout 자체는 비정상 종료 코드에도 stdout을
전달한다. 종료 코드를 검사하는 save_success는 별도 학습 기능이다.
연결과 성공 분기를 한 번에 수행하는 전체 end-to-end 작업은 아직 배우지 않았다.

최종 근거: `results/process_chain_final_verified`의 curriculum/program/summary.
관련 테스트 19개(25.08초), 전체 회귀 286개(251.19초), 안내 변경 후 어댑터
6개(6.04초)가 통과했다.
이전 `process_chain_verified`, `process_chain_branched_verified`는 중간 통과 기록,
`process_chain_guarded_verified`, `process_chain_composed_verified`는 EXCLUDED 기록이다.
이는 측정된 사례 성공이며 범용 정확성이나 관찰 기록의 독립 인증이 아니다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_process_chain.py tests/test_stateful_control.py tests/test_stateful_deadline.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.process_chain --output results/process_chain_replay
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

다음은 동시 스트리밍 파이프, 프로세스 파일 변경의 관찰·학습, 네트워크와 시스템
정보다. Windows/macOS 프로세스는 미검증이며 컨테이너는 기존 하나만 사용한다.
