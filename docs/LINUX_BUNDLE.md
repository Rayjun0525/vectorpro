# 제한된 리눅스 기능 통합

2026-10-05. 기존 프로그램 파일에 다섯 기능을 누적했다. 알려진 계약은 LLM 없이
호출하며, 예제에서 호출 순서와 인자 연결을 찾아 텐서에 저장한다.

| 습득 기능 | 발견한 조합 | 탐색 후보 |
| --- | --- | --- |
| pipeline_save | process.pipeline → 기존 save_success | 116 |
| created_content | process.run → file.read | 26 |
| http_checked | network.http → network.require_body | 6 |
| system_text | system.info → json.text | 6 |
| download_transform | http_checked → pipeline_save | 384 |

각 기능은 학습 2건과 별도 검증 2건을 사용했다. 저장 후 다시 로드한 파일에서
미사용 입력으로 실제 리눅스 검증 13건을 통과했다: HTTP→파이프→파일 4건,
HTTP 실패/파이프 실패 2건, 프로세스 파일·빈 디렉터리 생성 후 읽기 4건,
시스템 문자열 조회 3건. 빈 값, 바이너리, 한글, 8192바이트 입력을 포함한다.
외부 명령이 파일을 만드는 알고리즘 자체를 학습한 것은 아니다.

## 제공하는 실행 기능

process.pipeline은 1~8개 argv 명령을 실제 OS 파이프로 연결해 동시에 실행한다.
요청은 stages/timeout/pipefail JSON이다. shell을 사용하지 않는다. 결과는 기존
stdout/stderr/code에 단계별 codes를 추가한다. 기본 pipefail=true는 처음 실패한
단계의 코드를 반환하고 false는 마지막 단계 코드를 반환한다. stderr는 단계 순서로
합친다. 시간 한도는 전체 요청과 각 단계 설정 중 최솟값, 최대 30초다. 출력 한도와
실패 시 프로세스 그룹 정리를 제공한다. 요청의 argv와 파이프 단계 구성은 호출자가
제공한다. 학습 대상은 이 기능과 다른 기능의 연결이다.

network.http는 HTTP/HTTPS 요청과 상태/바이너리 본문을 제공한다. 상태 오류도 결과로
반환한다. network.require_body의 비2xx 예외 정책은 제공하는 범용 기능이다.
HTTP 성공 판정을 별도로 학습했다고 표현하지 않는다. 이번 실제 검증은 로컬 HTTP
서버만 사용했고 인터넷/TLS는 검증하지 않았다. urllib timeout은 I/O 제한이며 전체
HTTP 요청의 엄격한 총시간 제한은 아니다. 응답 크기를 제한한다.

system.info는 platform/machine/cpu_count/cwd/pid를 JSON으로 반환한다. 습득한
system_text는 문자열 조회 조합이며 숫자 조회는 호스트 기능 테스트로 확인했다.
권한 관리나 시스템 설정 변경은 포함하지 않는다.

## 학습 관찰 기록

processes는 operation(기본 process.run), request/stdin/stdout/stderr의 hex와 code를
기록한다. pipeline 기록은 합산 결과만 재생하며 단계별 codes는 실제 실행에서만
반환한다. 프로세스가 상태를 바꾸는 기록은 before/before_directories와
after/after_directories 네 필드가 모두 필요하다. 실행 직전 전체 상태가 before와
일치할 때만 after로 전이하므로 후보가 먼저 만든 불필요한 파일을 덮어 숨길 수 없다.
최종 목표도 전체 파일과 디렉터리 상태로 비교한다.

observations는 operation/request/response hex를 기록한다. network.http와 system.info만
지원한다. 각 큐는 순서대로 정확한 요청을 소비해야 하고 모든 기록이 소비되어야 한다.
후보 탐색은 MemoryHostContext 안에서만 수행한다. 학습 테스트는 Popen/urlopen/실제
시스템 조회를 금지한 상태에서 성공했다. 사례당 외부 기록 총 16개, 기록 바이트
65536, 상태별 파일/디렉터리 각 32개로 제한한다.

## 재현과 증거

기존 vectorpro-test 컨테이너에서 실행한다. 결과 경로는 새 경로여야 한다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_linux_bundle.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.linux_bundle --output results/linux_bundle_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

최종 증거는 results/linux_bundle_final_verified의 curriculum.json/learning.json/
program.pt/summary.json이다. linux_bundle_verified는 실행 전 상태 검사를 강화하기 전
첫 실험 기록이며 최종 검증 근거는 final_verified를 사용한다. 파이프의 동시성은
생산자가 소비자의 신호 파일을 기다리는 테스트로 별도 확인했다. 순차 버퍼 실행은
이 테스트를 통과할 수 없다. 시간 초과/중간 실행 오류/출력 한도에서 자식 회수도 검사한다.

## 현재 단계와 한계

파일·디렉터리·프로세스·입출력·기본 HTTP·시스템 조회를 조합하는 제한된 초기모델의
기반을 구현하고 검증했다. 저장 형식/기존 계약 ID를 유지한다. 측정된 사례 성공이며
임의의 리눅스 프로그램을 대체하는 범용 정확성 증명은 아니다.

남은 과제는 자연어 요청에서 독립적인 목표·증거를 준비하는 과정의 신뢰성,
더 긴 작업과 동시 상태 변경, 권한/링크/스트림의 폭넓은 의미론, 실제 인터넷/TLS,
재시도·복구 정책과 추가 OS 검증이다. 임의 프로세스는 OS 샌드박스가 아니며 실패 시
외부 명령이 이미 변경한 파일을 자동 롤백하지 않는다. macOS 검증은 사용자 요청대로
후속 과제로 남긴다.
