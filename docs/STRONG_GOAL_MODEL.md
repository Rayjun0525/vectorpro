# 로컬 Ollama Gemma 12B 비교

2026-10-05. 사용자가 기존 nerdctl 규칙 대신 Docker 컨테이너 하나를 새로 만들고
그 안에서만 테스트하도록 명시했다. `vectorpro-strong-test` 하나를 생성해 모든
모델 요청·pytest·가상/실제 파일 실행·결과 집계를 그 컨테이너에서 수행했다.
추론 서버는 사용자가 이미 실행 중인 호스트 Ollama다. 컨테이너와 설치 의존성은
재사용할 수 있도록 남겼고 이미지 빌드/commit, 다른 컨테이너 시작, 모델 다운로드는 없다.

## 모델과 환경

- 설치 모델 `gemma4:12b`, 11.9B, Q4_K_M, 약 7.56GB.
- Ollama 모델 digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
- 모델 상세와 Ollama 버전은 `results/strong_goal_gemma4_12b/model.json`에 보존했다.
- 기존 이미지 `nousresearch/hermes-agent:latest`, 이미지 ID
  `sha256:24f49688cda7315ad39a9b94ef46347f3747e99863d5e05b5b86ff794baefdb3`.
  기존 Python이 있는 이미지여서 사용했으며 에이전트 진입점은 실행하지 않았다.
- Linux aarch64, Python 3.13.5, torch 2.14.1+cpu, numpy 2.5.3, pytest 9.1.1.
  의존성은 새 컨테이너 내부에만 uv로 설치했다.
- `/workspace`에 프로젝트를 마운트하고 `PYTHONPATH=/workspace/src`,
  OMP/MKL 스레드 1을 사용했다. Ollama 주소는 `http://host.docker.internal:11434`.

## 비교 방법

직전 시험의 16건을 재시험하고, 별도 영어/한국어 요청 16건을 이번 시험 전에
고정했다. 각각 지원 목표12건과 미지원/모순/모호4건이다. 새 요청은 실험 작성자가
작성한 작은 표본이며 실제 사용자의 독립 검토 이력이나 범용 자연어 평가가 아니다.
원문과 정답은 첫 모델 추론 전에 `cases.json`에 저장했다. 모델에는 원문·인터페이스·
상태 의미·도구 schema만 제공하고 기대 조건, 어시스턴트 답안, 교사 예제는 보내지 않았다.
결과를 보고 프롬프트/정답/모델 옵션을 바꾸거나 재시도하지 않았다.

기존 `propose_goal(..., encoding='rules')`와 기존 `compact_model_goal`을 그대로
사용한다. Ollama `/api/chat`의 schema `format`으로 한 도구 JSON envelope를 제약한다.
기존 1B도 JSON 문법 제약을 썼지만 llama.cpp의 tool 태그와 Gemma 템플릿을 사용했다.
이번 Ollama는 도구 목록과 JSON envelope 지시를 system에 덧붙인다. temperature0,
seed0, num_predict600, num_ctx8192, think=false. Ollama가 실제 반환한 메시지와
시간/토큰/종료 사유를 `raw_calls.json`에 그대로 보존했다.

큰 모델에 정답 목표를 대신 입력하지 않았다. 모델이 제안한 목표만으로 기존 복사/이동
계약의 정방향/역방향 연결을 새 메모리 호스트 두 입력에서 검사했다. 유일한 경로일 때
결과 폴더 안의 실제 리눅스 파일에 세 입력(빈 값/바이너리/한국어)을 실행했다.
정답 목표는 경로 선택에 사용하지 않고 마지막 전체 파일/디렉터리 상태 채점에만 썼다.
실행은 실험용 격리 폴더에서만 수행했으며 실제 사용자 요청에 대한 승인으로 취급하지 않는다.
역할-경로는 호출자가 `source.bin`/`target.bin`으로 제공했다. 임의 경로 추출은 평가하지 않았다.

## 결과

정확도는 기존 시험과 같은 정규화된 조건 배열의 **엄격한 일치** 기준이다.

| 모델/시험 | 형식 | 전체 일치 | 지원 목표 일치 | 불일치 초안 |
|---|---|---:|---:|---:|
| 이전 Gemma3 1B Q8_0, 같은16건 | rules | 2/16 | 0/12 | 0 |
| 이전 Gemma3 1B Q8_0, 같은16건 | compact | 4/16 | 0/12 | 1 |
| Ollama Gemma4 12B, 같은16건 | rules | 4/16 | 0/12 | 0 |
| Ollama Gemma4 12B, 같은16건 | compact | 13/16 | 9/12 | 2 |
| Ollama Gemma4 12B, 새16건 | rules | 3/16 | 0/12 | 1 |
| Ollama Gemma4 12B, 새16건 | compact | 14/16 | 10/12 | 1 |

compact의 불일치 초안3건은 원본 보존을 `unchanged` 대신 `equals_initial:self`로
표현했다. 해당 파일 입력에서 실제 최종 상태는 모두 정답과 일치했다. 이들을 의미상
잘못된 복사/이동으로 단정하지 않으며 엄격한 일치 점수도 사후 수정하지 않는다.
rules의 새 전체 보존 초안도 모든 항목 unchanged와 넓은 frame을 함께 반환한
동등 표현이지만 엄격한 일치에서는 제외됐다. 모든 논리적 동등성을 증명하는 평가가 아니다.

rules는 대부분 필수 frame 조건을 빠뜨려 거절됐다. compact는 기존 역방향 복사
1건과 새 한국어 전체 보존1건을 보류했다. 새 평가의 미지원4건은 모두 보류했다.
보존/삭제 목표를 맞춘 사례도 이번 저장 파일에 그 계약이 없어 실제 실행하지 않았다.

모델 목표로 선택한 저장 텐서의 native 실행은 기존9요청×3입력 **27/27**,
새8요청×3입력 **24/24**, 총51회 호출에서 전체 파일/빈 폴더 상태가 일치했다.
51개 독립 작업을 배운 것이 아니며 복사/이동 두 계약의 여러 요청·입력 검증이다.
원래 프로그램 SHA-256 `277964b962ab167e840b9f8f21544577696d14548ecf8b3da203d80582438f2c`
및 바이트가 유지됐고 학습/기능 등록/기본 경로 변경은 없다.

각16회 호출의 벽시계 합: 재시험 rules139.78초, compact87.62초,
새 rules138.61초, compact87.61초. 전체64회 **453.62초**.
첫 모델 로딩, 캐시, 서로 다른 추론 서버/하드웨어가 포함돼 1B와 속도 인과 비교를 하지 않는다.
정확한 토큰과 시간은 `metrics.json`에 있다.

## 검증과 재현

관련 **40 passed (1.97초)**. Ollama 응답 파싱/정답 미전달/불완전 JSON과
계약 부재 시 native 폴더를 만들지 않는 경계를 포함한다. 런타임/학습기/저장 형식은
바꾸지 않았으므로 전체 회귀는 재실행하지 않았다. 직전 전체398 passed 기록과 구분한다.

```sh
docker exec -e PYTHONPATH=/workspace/src -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-strong-test python -m experiments.strong_goal_model --output results/strong_goal_gemma4_12b_recheck
docker exec -e PYTHONPATH=/workspace/src -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-strong-test python -m pytest -q tests/test_strong_goal_model.py tests/test_assistant_intercept.py tests/test_goal_evidence.py tests/test_goal_draft.py tests/test_verified_reuse.py
```

기존 결과를 덮어쓰지 않는 새 출력 경로를 지정한다. 결과 원본은
`results/strong_goal_gemma4_12b/{cases,model,raw_calls,summary,metrics}.json`이다.

## 판단과 남은 과제

실제 큰 모델에서 간결한 목표 형식의 개선을 확인했다. 모델이 커도 복잡한 조건
작성 형식은 실패하므로 크기 하나만으로 설명하지 않는다. 모델 세대/양자화/템플릿/
추론 백엔드도 달라 모델 크기만의 통제 비교가 아니다. 새16건에는 같은 백엔드에서
실행한 1B 대조군도 없다. 이번 결과를 기본 실행의 자동 승인 기준으로 채택하지 않았다.

현재 단계는 **큰 모델의 목표 해석 개선과 저장 텐서 연결을 실측한 단계**다.
다음은 실제 경로/역할 추출, 목표 표현의 의미상 동등성, 지원 범위 확대와 더 다양한
새 요청의 독립 평가, 같은 백엔드/세대에서 크기만 바꾼 대조군이다.
