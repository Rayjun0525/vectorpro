# 프로세스 호출과 결과 전달

`process.run`의 제공 의미와 예제에서 습득하는 호출 절차를 구분한다.
이번 범위는 동기 프로세스 하나의 argv/stdin/cwd/env/시간 제한, stdout/stderr와
종료 코드, 학습된 stdout 저장 절차다. 외부 실행 파일의 내부 알고리즘을 학습하거나
그 실행 파일을 텐서로 대체한 것은 아니다.

## 실행 API

opt-in `process.run(buffer request, buffer stdin) -> buffer result`.
request는 UTF-8 JSON `{argv:[executable,arg,...],cwd?:relative_path,env?:{key:value},timeout?:seconds}`다.
shell은 사용하지 않고 argv를 직접 전달한다. cwd를 생략하면 실행 때 제공한
호스트 루트를 사용하며 지정 경로·심볼릭 링크는 루트 안이어야 한다.
env는 실행 환경에 명시한 문자열을 덮어쓴다. timeout은 기본 5초, 최대 30초다.

결과 버퍼는 JSON `{stdout:hex,stderr:hex,code:signed_integer}`다. 코드와 스트림을
함께 보존하고 `process.stdout/stderr -> buffer`, `process.code -> value`로 꺼낸다.
음수 POSIX 신호 종료는 숫자 레지스터에서 `128 + signal`로 반환한다.
실행 파일 없음, 잘못된 요청, 시간 초과·출력 한도는 오류로 반환한다.
비정상 종료 코드는 결과이며 stderr를 버리지 않는다.

출력은 임시 파일로 받고 20ms 간격으로 한도를 확인해 무제한 RAM 적재를 피한다.
스트림 원문 합은 max_buffer_bytes/2 이하여야 하고 hex 결과 전체에도 버퍼 한도를
적용한다. 이는 OS의 강제 디스크 할당량이 아니어서 검사 사이 쓰기는 초과할 수 있다.
POSIX 실행은 새 프로세스 그룹을 만들고 성공/실패 모두 같은 그룹을 정리한다.
호스트 cwd 확인은 OS 권한 샌드박스가 아니다. 실행 파일은 컨테이너의 권한으로
다른 경로·환경에 접근할 수 있다. 기존 vectorpro-test 하나에서만 검증했다.
Windows 프로세스/자식 트리, 백그라운드 daemon 분리, macOS는 검증하지 않았다.

## 학습의 격리와 관찰 근거

상태 사례에 `processes:[{request:hex,stdin:hex,stdout:hex,stderr:hex,code:int}]`를 넣는다.
MemoryHostContext는 순서대로 정확한 요청/입력에 해당하는 기록만 재생한다.
다른 요청이나 초과 호출은 실패하고 모든 기록을 소비해야 한다. subprocess 호출은
학습에서 금지한다. 테스트에서 Popen을 실패 함수로 바꿔 이 경계를 확인했다.
기록은 정답 근거이며 실제 프로세스 관찰의 신뢰성을 독립 인증하는 기능은 없다.
현 범위는 프로세스 자체의 파일 변경을 모델링하지 않고 stdout을 호스트 쓰기로 연결한다.

학습·검증에 tr의 대문자 출력 기록을 사용해 stdin 그대로 저장하는 우회 후보와
stdout을 추출하는 후보를 구분했다. tr의 동작은 제공된 외부 프로그램이며 예제
정답 생성 코드가 native 실행에 참여하지 않는다. 실제 tr의 입력/출력도 검사했다.
습득 절차는 `process.run → process.stdout → file.write` 3호출이며 인자 경로와
순서를 상태 예제에서 선택해 제어 텐서에 저장한다. 1340후보에서 채택했다.
배운 경로는 이전 record_results 텐서 파일에 축적했다.

새 미사용 native 요청은 tr 대신 cat으로 바꾸고 빈/바이너리/한글/4096바이트를
처리했다. 같은 텐서를 저장·재로딩해 정확한 파일 상태와 출력 길이 **4/4** 통과했다.
관련 회귀 11개(70.06초), 최종 전체 회귀 281개(243.35초)가 통과했다.
LLM 없이 직접 계약을 호출했다. 종료 7, stderr, cwd/env, 신호 종료, 실행 파일
없음, 시간 제한, 출력 제한과 잘못된 검증의 등록 실패도 테스트했다.

프로세스 관찰을 사용하는 순차 후보는 버퍼 생산자를 먼저 시도하고 일반 명령
메모리 시뮬레이션으로 학습 불일치를 거른다. 기존 일반 제어 탐색은 유지한다.
채택 후보는 실제 컴파일된 텐서로 학습·검증을 재확인한다.
후보 20000개·시간 60초 제한은 유지했다. 예비 시도의 시간/후보 한도 실패도 보존한다.
자료: `results/process_results_verified_ordered`가 채택된 결과이며
`process_results_verified`, `process_results_verified_final`은 제외된 예비 시도다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_process_results.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/process_results.py --output results/process_results_replay
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

현재 단계는 리눅스 프로세스 입출력의 초기 연결이다. 남은 것은 결과를 다른
프로세스 stdin으로 이어가는 학습·검증, 동시 OS 파이프, 프로세스 실패 분기,
파일 변경 관찰·네트워크·시스템 정보다. 전체 프로그램의 범용 정확성 증명은 아니다.
이후 순차 연결과 종료 코드 저장 분기까지 검증했다. 최신 범위는
[PROCESS_CHAIN.md](PROCESS_CHAIN.md)를 참고한다. 동시 OS 파이프는 남아 있다.
