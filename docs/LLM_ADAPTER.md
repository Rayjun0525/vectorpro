# LLM 학습 어댑터

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
