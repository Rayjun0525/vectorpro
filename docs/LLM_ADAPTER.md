# LLM 학습 어댑터

현재 기준은 사용자 요청으로 복원한 Gemma 3 1B Instruct Q8_0이다.
Laya 가중치는 제거하고 시험 기록은 [LAYA.md](LAYA.md)에 보존했다.
명시적 폭 선택지 제한과 재현 명령은 [GEMMA_BASELINE.md](GEMMA_BASELINE.md)에 있다.
로컬 runner는 Gemma를 기본값으로 사용하며 자동 다운로드하지 않는다.

## 후속: 요청 경로의 원문 보존

경로 인자는 카탈로그 계약의 `path` 타입에서 식별한다. 원래 요청에서 따옴표로
묶은 문자열과 확장자가 있는 경로 토큰을 추출해 도구 선택지에 넣고, 백엔드도
값을 검사한다. 한글 조사 뒤의 파일명, 공백이 있는 따옴표 경로, 여러 점/하위
디렉터리를 보존한다. 따옴표 경로 내부의 짧은 조각을 별도 경로로 추가하지 않는다.
가상 예제/디렉터리 목록에서 실제 경로를 가져오지 않는다. 파일명 안의 숫자는
숫자 인자 선택지에서 제외한다. 경로 리터럴이 없으면 경로 준비 도구를 제공하지
않고 누락 정보를 질문하도록 한다. 직접 이름으로 실행하는 런타임 경로는 유지한다.

이 장치는 원문에 없는 `IN` 같은 문자열을 막지만 여러 경로 중 어느 것이 대상인지
증명하지 않는다. 따옴표 없는 확장자 없는 경로, 복잡한 자연어 경계는 미지원일 수
있으며 따옴표의 일반 문자열도 선택지가 될 수 있다. OS 경로 정규화/호스트 루트
제약과 실제 실행 오류 처리도 그대로 유지한다. 학습은 메모리에서만 수행한다.

회귀 186건(280.11초), 관련 32건(9.90초) 통과. 기존 한국어 실패 재검증은 1/1,
새 첫 평가 6건은 전체 4/6이다. 실행 대상 5건 모두 경로가 일치하고 경로 누락
요청은 질문했다. 한국어 채우기 2건은 경로가 맞아도 값을 폭으로, 폭을 값으로
준비해 실패했다(29/31비트, 채울 값 16). 실제 테스트 대상 파일이 바뀌었으며
보존 파일은 그대로였다. 경로 선택지의 정확성과 작업 전체의 정확성을 구분한다.
평가 후 재튜닝하지 않았다. 원문과 SHA는 `results/gemma_paths/`에 보존한다.

### 자동 학습의 정답 근거 점검

현재 `ExampleLesson.validate`는 타입/폭/중복/학습·검증 분리를 검사하고, 학습기는
주어진 정답에 대한 후보 일치를 검사한다. 같은 LLM이 만든 서로 다른 사례는
입력 분리일 뿐 정답 출처의 독립 검증이 아니다. 잘못된 정답이 일관되면 의도와
다른 기능을 배울 수 있다. 사례 성공과 일반화 또는 자연어 목표의 정확성을 구분한다.

다음 학습 단계에는 독립적인 근거 출처가 필요하다. 예를 들어 기존 실행 프로그램의
관찰 결과, 사람이 승인한 목표 파일 스냅샷, 별도 검증기가 제공한 데이터 사례다.
LLM은 입력/학습 계획을 준비하고 근거 출처가 정답을 제공하도록 연결할 수 있다.
작업별 정답 알고리즘을 커널에 추가하거나 기존 후보 출력으로 정답을 만들어서는
의도 검증이 되지 않는다. 현재 코드의 `TargetSource`는 이런 출처의 인터페이스지만
임의 외부 프로그램/검증기를 자동으로 연결한 구현까지 완료한 것은 아니다.

실제 Gemma NAND 진단은 경로 대신 버퍼 입력을 선택해 학습 단계에 도달하지
못했다(0/1). 기능 등록/별도 100건 평가는 없었다. 정답 생성 이전에 요청의 입력
타입과 숫자 역할을 확정하는 단계도 필요하다. 이 진단은 이전 개발 요청의 재현이다.

## 후속: 실제 입력 분리와 실행 완료 반환

카탈로그 요청의 `prepare_N`은 원래 요청과 인자 계약만 담은 별도 모델 문맥에서
준비한다. 검색 이력이나 가상 예제를 실제 입력으로 복사하지 않는다. 요청에
숫자 리터럴이 있으면 숫자 인자의 선택지를 그 리터럴로 제한하고 백엔드도 검사한다.
계산 결과나 기본 0을 임의로 입력으로 만들지 않는다. 폭 선택지는 1..32 안의
리터럴이며 해당 값이 없을 때 16을 제공한다. 숫자가 어떤 역할인지 증명하는
장치는 아니며 파일명 속 숫자, 자연어로 쓴 수와 복잡한 수 표현은 한계가 남는다.

구조화된 caller 근거가 있으면 실제 인자를 준비한 직후 백엔드가 원본 근거를
기존 격리 검증기에 전달한다. LLM이 예제를 다시 작성하지 않는다. 유일한 후보가
검증돼야 `execute_resolved`가 제공되며 모호한 근거/실패는 실행하지 않는다.
caller 근거가 없는 숫자 예제는 여전히 모델이 작성한다. 상태 요청에는 기존의
구조화된 caller 근거 요구를 유지한다. 작업별 알고리즘이나 정답 생성은 추가하지 않았다.

카탈로그 실행이 완료되면 모델의 추가 턴 없이 `status: executed`, 실제 `outputs`,
`capability`, `request`(query/width/operands), 도구 기록을 반환한다. CLI 종료 코드는
0이다. 이 완료 반환은 LLM이 쓴 성공 문장이 아니라 실행기의 결과다. 기존 기본
카탈로그 없는 경로의 `answered` 응답은 유지한다.

새 평가 파일 `experiments/requests/gemma_binding_first_use.json`은 출력과 전체 파일
스냅샷 외에 실제 폭/입력, 완료 상태를 검사한다. 기대 인자는 평가용이며 모델에
전달하지 않는다. 이전 7/12의 출력 기준과 더 엄격한 새 기준을 직접 비교하지 않는다.

최종 회귀 183건(338.79초), 관련 29건(17.28초) 통과. 새 첫 평가 8건은 7/8:
숫자 4/4, 파일 1/2, 미지원 1/1, 모호한 요청 1/1이다. 한국어 경로 `결과.bin`을
`IN`으로 잘못 준비해 실제 실행에서 파일 조회 오류가 났고 원본을 유지했다.
이 평가를 본 뒤 수정하지 않았다. `results/gemma_binding/first-use`의 원문과
`verification.json`의 소스/데이터 SHA를 보존한다. caller 근거를 받은 실행 흐름의
유한 사례 검증이며 자동 정답 생성/새 기능 학습 또는 범용 정확성 증명이 아니다.

## 현재 모델: Gemma 3 1B

사용자 요청으로 컨테이너의 Qwen3-0.6B 가중치를 삭제하고 Gemma 3 1B Instruct
Q8_0으로 교체했다. 모델은 `/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf`에 있다.
배포 저장소 `ggml-org/gemma-3-1b-it-GGUF`, revision
`f9c28bcd85737ffc5aef028638d3341d49869c27`, 1,069,306,368바이트,
SHA-256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`.
기존 `vectorpro-test`와 llama-cpp-python 0.3.36을 재사용했다.

Gemma의 원래 대화 템플릿을 사용하며 도구 JSON 스키마와 호출 형식을 메시지에
명시한다. system/tool 역할을 user 메시지로 표현하고 assistant 도구 호출은
`<tool_call>` JSON으로 전달한다. 연속 메시지만 합치고 실제 입력·caller 근거·
도구 결과 값은 유지한다. 모델이 도구/인자를 직접 선택하며 스크립트 답변은 없다.
BOS를 한 번만 넣고 Gemma 종료 토큰을 사용한다. 알려진 벡터 기능은 계속 LLM 없이
실행되고 MiniLM 색인과 벡터 저장 형식은 유지한다. 아래 Qwen 기록은 교체 전 결과다.

```powershell
nerdctl exec vectorpro-test python experiments/setup_gemma_model.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/gemma_model/reproduction --constrain-tools --evaluation experiments/requests/catalog_agent_holdout_v2.json --max-calls 12
```

고정된 기존 요청의 모델 교체 후 재현이며 새 독립 평가로 표현하지 않는다.
Qwen 제거/설치 기록은 `results/gemma_model/replacement.json`에 있다.

교체 후 전체 회귀는 181건 통과(374.36초), 관련 프로토콜/어댑터 테스트는
27건 통과(10.14초)다. Gemma 실제 모델 결과의 `passed`는 기존 평가기의
실행 출력/파일/실패 격리 기준이다. 실행 후 추가 질문이나 잘못된 폭 해석까지
모두 검증한 완전한 대화 정확도가 아니다. 실제 영어 XOR는 출력 99가 맞아도
8비트 요청을 32비트로 준비했고, 실행 뒤 불필요한 호출/질문으로 종료했다.
이런 한계를 성공 숫자와 함께 기록한다. 자체 예제 생성 문제도 모델 교체만으로
해결됐다고 표현하지 않는다.

실제 Gemma 재현 결과는 **7/12**(`results/gemma_model/catalog-v2`). 숫자 출력
3/4, 파일 변환 0/2, 미지원 3/3, 모호한 요청 질문 1/1, 자체 예제/학습 0/2다.
영어 뺄셈은 실제 인자를 잘못 준비했다. 파일 작업은 잘못된 경로/값과 중복
스냅샷을 제출하고 호출 예산으로 종료했으며 두 파일 모두 원본을 유지했다.
자체 XOR는 잘못된 정답으로 거절됐다. NAND는 숫자 요청을 파일 요청으로
해석해 질문으로 종료했고 기능 등록/별도 100건 평가는 수행되지 않았다.
결과·원문과 설치 기록은 `results/gemma_model/`에 보존한다.

현재 재현의 원시 summary `evaluation_protocol`은 실행 시작 당시 코드가 읽은
기존 데이터셋의 첫 평가 라벨이다. 이번 실행은 재현이다. 해석은
`verification.json`의 `run_semantics`를 따른다. 이후 스크립트는 데이터셋 라벨을
`dataset_protocol`로 분리하고 모델 교체 재현임을 별도로 표시한다.
모델 교체는 완료됐으며 남은 과제는 정확한 요청 인자 해석, 실행 후 답변 종료,
정확한 자체 예제와 새 기능 학습, 새 독립 요청 평가 및 외부 서버 실제 검증이다.

선택적인 텐서 카탈로그 경로는 [CATALOG_AGENT.md](CATALOG_AGENT.md) 참고.
`--encoder`, `--no-learning`, `--evidence-file`을 추가했고, 실제 입력을 먼저
고정한 뒤 검증 예제를 전달한다. 아래의 기존 기본 도구 경로도 유지한다.

후속 구현은 근거 선검사, 행 단위 `check_small`/`teach_small`, 실제 숫자를 기호로
가린 별도 예제 준비 문맥을 포함한다. caller 근거와 모델 자체 정답을 구분하고,
정답을 기존 후보의 출력으로 대신 만들지 않는다. 상세는 CATALOG_AGENT.md 참고.

LLM은 사람의 요구를 목표·예제로 바꾸고 학습기의 도구를 사용하는 역할이다.

카탈로그 LLM의 파일/상태 요청은 caller가 제공한 구조화된 `state_validation`이
있어야 실제 실행을 승인한다. 모델이 작성한 파일 스냅샷만으로 승인하지 않는다.
새 요청 첫 평가 9/12와 수정 후 동일 파일 사례 1/1 회귀 확인은 구분한다.
모델의 자체 정답 생성·새 기능 학습은 아직 안정적으로 성공하지 않았다.
실제 작업은 학습된 벡터 프로그램이 실행하며, 프로그램 파일에는 LLM이나
Python 작업 코드가 들어가지 않는다.

## 실행

Python/PyTorch가 설치된 환경에서 `python -m vectorpro.agent`를 사용한다.
설치 후에는 `vectorpro-agent` 명령도 제공한다. 프로젝트의 테스트는 항상
기존 `vectorpro-test` 컨테이너 안에서 진행한다.

```text
python -m vectorpro.agent --program program.json --host-root data --endpoint http://your-model-host:port/v1/chat/completions --model YOUR_MODEL --intent "요청 내용"
```

주소는 `/chat/completions`까지 포함하는 전체 주소이고, 모델은 호출자가
선택한다. 인증이 필요하면 `VECTORPRO_LLM_API_KEY` 환경변수를 설정한다.
`--api-key-env`로 다른 환경변수 이름을 지정할 수 있다. 키는 프로그램 파일에
저장하지 않는다. 컨테이너의 localhost는 호스트 PC의 localhost와 다르므로
연결 가능한 주소를 지정한다.

함수 호출 메시지와 도구 결과 반환 형식은
[OpenAI 공식 함수 호출 문서](https://developers.openai.com/api/docs/guides/function-calling)를
따른다. 사용할 엔드포인트와 모델은 해당 함수 호출 방식을 지원해야 한다.
외부 API의 실제 호출은 주소·모델·인증 설정 후 수행한다.

## 도구와 흐름

시스템 지침에 학습 형식과 한계를 제공한다. `list_capabilities`는 배운 기능과
제공된 OS 기능을 구분하고, `describe_capability`는 저장된 텐서의 실제 절차를
설명한다. `teach`는 숫자 또는 상태 예제를 받아 가상 환경에서 학습하고 저장한다.
`execute`는 이미 알려진 기능에 새로운 입력을 전달한다. 모호한 목표는
`ask_user`로 질문을 반환한다. 라이브러리의 `AgentSession`을 유지하면 사용자
답변을 넣어 대화를 이어갈 수 있다.

학습 단계는 실제 파일을 실행 대상으로 삼지 않는다. 실제 실행의 파일 루트는
호출자가 `--host-root`로 지정하며, LLM이 임의로 바꾸지 못한다. 파일 경로는
UTF-8 상대 경로로 전달하고 실행 때 새 핸들을 할당한다.

한 응답에 도구 한 개를 순차 호출하며, 기본 최대 12번의 모델 턴을 허용한다.
상태 학습은 60초·후보 20,000개·호출 8개·실행 clock 20,000개 이하로 제한한다.
숫자 예제도 입력 수·비트 폭·사례 수를 제한한다. 코드·바이너리·텐서 명령을
학습 입력에 직접 넣는 방식은 허용하지 않는다.

## 정답의 근거

LLM이 만든 정답도 틀릴 수 있다. 학습·검증 사례를 나누는 것은 같은 예제의
암기를 줄이지만, 모델이 사용자의 의도를 제대로 이해했다는 독립 증명은 아니다.
가능하면 사람이 확인한 사례나 기존 시스템의 관측값을 제공한다. 어댑터가
제공한 예제는 프로그램 provenance에 출처를 기록한다.

현재 자동 테스트는 스크립트로 정한 모델 응답과 컨테이너 안의 HTTP 서버를
사용한다. 실제 네트워크 전송·도구 결과 재입력·학습·실행·저장·질문·실패를
검증하지만, 실제 LLM의 자연어 이해 능력을 측정한 결과로 취급하지 않는다.

## 어시스턴트의 중간 도구 호출 검증

사용자의 허용으로 현재 어시스턴트가 예제와 다음 도구 호출을 직접 결정하고
`scripts/intercept_tool.py`를 통해 실제 학습기에 전달하는 테스트도 수행했다.
외부 모델 API 응답을 흉내 낸 것이 아니며, 외부 API 호출도 하지 않았다.
NAND 요청에서 미지 요청 → 예제 학습 → 저장 → 새로운 입력 실행을 확인했고,
16/32비트 미사용 입력 100개 및 Windows 실제 파일 변환이 성공했다.
결과는 `results/initial_model/interception-summary.json`에 있다.

바탕화면 테스트 폴더의 `01-list.json`부터 `05-file.json`까지는 실제 도구 결과다.
`assistant-program.json`에는 새 기능이 누적됐고, 원래 `program.json`은 보존했다.
외부 모델 자동 호출 품질이나 임의의 자연어 요청 전반이 검증된 것은 아니다.

## 실제 소형 모델 검증 (2026-10-04)

기존 `vectorpro-test` 안에 `llama-cpp-python==0.3.36`과 공식
`Qwen/Qwen3-0.6B-GGUF`의 Q8_0 파일(639,446,688바이트)을 설치했다.
모델은 `/opt/vectorpro-models/Qwen3-0.6B-Q8_0.gguf`에 보관하고 새 이미지나
컨테이너는 만들지 않았다. 모델 revision은
`23749fefcc72300e3a2ad315e1317431b06b590a`다.

`experiments/local_small_llm.py`는 실제 CPU 추론 → 로컬 HTTP → 기존
`HTTPChatModel`/`AgentSession` → 실제 도구 실행을 연결한다. 모델의 도구 선택과
인자는 대체하지 않는다. Qwen 템플릿에 전달할 때 null content를 빈 문자열로
변환하는 연결 수정 후, 영어 요청 4건 중 엄격한 성공 기준은 1건 통과했다.

- 알려진 XOR: 정상 호출과 결과 8686 확인.
- 모호한 요청: 말로 질문했으나 `ask_user`를 호출하지 않아 `needs_input` 상태 실패.
- 파일 map: 잘못된 함수 이름과 빈 입력으로 실행을 요청했고 오류 후 중단.
- 새 NAND 학습: 예제 형식이 잘못된 teach 요청 3회 후 중단. 기능 등록 없음.

결과와 원문 모델 출력은 `results/local_small_llm/verified/`에 보존한다.
초기 연결 실패는 상위 폴더 및 `diagnostic/`에 별도로 보존했다.
모델이 사용법을 자동으로 충분히 따를 것이라는 가정은 이 모델에서 성립하지
않았다. 다음 과제는 더 명확한 도구 스키마·단계별 안내·오류 회복의 검증이며,
다른 모델이나 한국어 요청에 대한 성공률은 측정하지 않았다.

재현 명령(모델과 의존성은 같은 컨테이너에 남아 있다):

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --root results/local_small_llm/recheck
```

이 테스트는 프로세스 안의 임시 HTTP 서버를 사용하고 종료 시 서버를 닫는다.
상시 서비스로 설치한 것은 아니다. [공식 모델](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF).

## 소형 모델을 위한 입력 간소화

`teach_numeric`는 name/description/output과 training/validation 사례만 받는다.
어댑터가 입력 수·사례 수·학습 폭으로 LearningPlan을 구성한다. 정답이나
작업 절차를 생성하지 않고, 기존 `teach`를 그대로 호출하므로 기존 호출도 호환된다.
각 요청에 현재 기능 목록을 제공하고, 문자열 경로/숫자 및 입력 수 오류에는
수정에 필요한 형식 안내를 반환한다. 누적 기능의 실행에는 LLM이 필요 없다.

실제 모델 비교 로그는 `optimized/`, `optimized_json/`, `optimized_recovery/`에 있다.
첫 안내의 Python식 예시는 모델이 실행 대신 문장을 출력하게 해 제거했다.
수정된 JSON 안내에서는 파일 map이 성공했으나 질문 프로토콜은 여전히 실패했다.
숫자 학습 도구 호출은 가능해졌지만 잘못 생성된 예제로 NAND 대신 다른 기능을
습득했고 독립 미사용 사례는 0/100이었다. 예제 일치만으로 의도 달성을 판단하지 않는다.

테스트 서버의 `--constrain-tools` 옵션은 llama.cpp JSON-schema 문법으로 도구
응답 형식을 제한한다. 모델이 함수 이름과 인자 값을 선택하며 정답은 공급하지 않는다.
실행 성공 이후에는 자연어 응답을 허용한다. HTTPChatModel 자체가 모든 외부
서버에서 이 제약을 지원하는 것은 아니며, 각 서버에서 별도로 설정해야 한다.
비교 명령:

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --root results/local_small_llm/constrained-recheck --constrain-tools
```

Gemma 비교 후보는 도구 호출용 [FunctionGemma 270M](https://ai.google.dev/gemma/docs/functiongemma)다.
아직 다운로드하거나 테스트하지 않았고, 일반 Gemma와 동일한 호출 템플릿으로
동작한다고 가정하지 않는다. macOS 검증은 사용자 결정으로 보류한다.

형식 제약 비교는 4건 중 3건 성공했다(`constrained/summary.json`). XOR 결과,
needs_input 질문 상태, 실제 파일 map 결과가 모두 일치했다. NAND 학습은 모델이
AND 정답을 만들었고 미사용 100건에서 0건 성공했다. 단일 사례로 이를 등록하는
문제를 줄이기 위해 새 teach_numeric 도구는 학습 최소 4건·검증 최소 2건을
요구한다. 이는 정답 정확성을 보장하지 않는다. 추가 결과는 `constrained_evidence/`.
최소 사례 수 적용 후 재검증에서도 모델이 중복 입력을 만들어 학습은 거절됐고,
ask_user로 필요한 예제를 요청했다. 이 경우 잘못된 기능은 등록되지 않았다.
전체 회귀 157건 통과(291.39초), 최종 어댑터 테스트 6건 통과(8.01초).

## 2026-10-05: 호출자 근거와 숨긴 사례를 통한 학습 채택

EvidenceBank 모드에서 LLM은 근거 ID와 새 이름을 선택하고, 백엔드가 고정된 인터페이스와 예제로 격리 학습한다. 학습에 제공하지 않은 사례를 통과하고 저장이 성공해야 등록한다. 승인 계약은 LLM 없이 직접 재사용할 수 있다. 실제 의도와 근거 출처를 자동 인증한 것은 아니다. 명세와 재현은 docs/VERIFIED_ACQUISITION.md, 최종 검증은 docs/HANDOFF.md를 참조한다.

## 2026-10-05: 설치된 외부 기준에서 근거 자동 수집

ReferenceProviders가 제공된 manifest의 실행 파일을 임시 루트에서 관찰해 학습/검증/숨긴 사례를 자동 생성한다. LLM은 기준 ID만 선택하고 실제 사용자 파일은 수집에 사용하지 않는다. 기준 설치와 의미 선택의 독립 검증은 여전히 남아 있다. 명세/재현은 docs/REFERENCE_EVIDENCE.md, 검증 결과는 docs/HANDOFF.md를 참조한다.
