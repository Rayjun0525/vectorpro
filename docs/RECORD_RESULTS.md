# 여러 결과 전달의 초기 기반

파일 읽기의 내용과 길이처럼 버퍼와 숫자를 한 번의 결과로 전달한다.
프로그램·계약·커널의 출력 레지스터는 기존처럼 하나이며, 레코드를 담은 버퍼를
반환한다. 레코드는 실제 데이터를 소유하므로 새 HostContext에서도 사용할 수 있다.
기존 저장 형식과 계약 버전은 유지하고 새 의미는 opt-in으로 설치한다.

## 제공 의미와 학습되는 절차

- `record.pack(buffer, value) -> buffer`: 버퍼와 unsigned 숫자 한 개를 묶는다.
- `record.buffer(buffer) -> buffer`: 원본 바이트를 새 버퍼로 꺼낸다.
- `record.value(buffer) -> value`: 숫자를 꺼낸다. 실행 폭에 맞지 않으면 실패한다.

이들은 범용 데이터 표현 의미다. 파일 읽기→길이 계산→레코드 생성 순서와
레코드에서 꺼내→쓰기 순서는 상태 예제에서 탐색하며 실행 커널에 내장하지 않는다.
이미 학습한 조건 선택·집계 파일에 두 기능을 축적했다.

이진 형식은 `VPR1` 4바이트, payload 길이 uint64 little endian 8바이트,
값 uint64 little endian 8바이트, 원본 payload 순이다. 전체 길이를 정확히 검사한다.
OS 핸들·경로를 고정하지 않으며 숫자·바이트를 담는다. 버퍼 자원 한도는
20바이트 헤더를 포함해 적용한다. 새 필드 이름이나 JSON 해석이 필요하지 않다.

호출자는 계약의 buffer 출력 `{hex: ...}`를 다음 계약의 buffer 입력으로 전달한다.
`record.value` 계약으로 숫자도 독립적으로 얻을 수 있다. 이것은 일반적인
이름 있는 필드·임의 튜플·중첩 레코드 체계의 완성이 아닌 두 결과 전달의 시작이다.

## 학습 근거와 검증

상태 예제에서 `output_type: "buffer"`와 `output: "<hex bytes>"`를 지원한다.
목표 출력은 핸들 번호가 아니라 실제 바이트로 비교한다. 기본은 기존 value 출력이다.
타입이 맞는 후보만 평가하며, buffer 반환 초안은 같은 output_type을 명시해야 한다.
파일/디렉터리 전체 상태도 그대로 비교한다. 현재 목록/버퍼 반복 문법은 value
반환만 지원하고 buffer 반환은 순차·기존 일반 제어 후보에서 탐색한다.

읽기 레코드: 300후보/3호출, 레코드 쓰기: 387후보/2호출.
각각 학습 2건·독립 검증 2건이며 실제 파일은 학습 중 건드리지 않았다.
저장·재로딩 후 별도 native 입력인 빈 데이터, 바이너리, 한글, 2048바이트
데이터에서 **4/4** 통과했다. 새 컨텍스트로 레코드를 전달해 숫자 추출과 쓰기를
확인하고 전체 파일 상태를 검사했다. 이 연결 호출 순서는 실험 시나리오이며
연결 전체를 하나의 학습 프로그램으로 발견했다는 뜻은 아니다.

잘못된 검증 출력은 등록 실패하고 기존 기능을 보존한다. 형식 오류, 길이 불일치,
자원 한도, 숫자 폭, 기존 계약 ID 재로딩도 테스트했다. 측정된 사례 성공은
일반적인 프로그램 정확성이나 인간 의도에 대한 증명이 아니다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_record_results.py tests/test_contract_learning.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/record_results.py --output results/record_results_replay
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

증거는 `results/record_results_verified`의 curriculum/summary/program에 보존한다.
관련 테스트 24개(11.27초), 전체 회귀 271개(211.62초)가 통과했다.
다음은 이 결과 기반 위에 프로세스 종료 코드와 stdout/stderr, 표준 입출력과
파이프를 연결하는 것이다. 일반 레코드 스키마 확장과 네트워크도 남아 있다.
