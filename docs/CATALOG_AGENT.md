# 텐서 파일과 LLM 요청 연결

2026-10-04, 기존 검색 프로토타입을 선택적인 런타임/LLM 어댑터로 연결했다.
기존 JSON 파일은 호환되고, `VectorRuntime.save/load`는 `.pt`도 지원한다.

## 한 파일에 저장하는 정보

- 실행 셀·호출·라우팅·제어 데이터와 학습 이력/RNG 상태의 정확한 타입 트리.
- 기능 설명·인자 타입/역할·효과·인코더 revision의 정확한 타입 트리.
- 의미 검색 벡터와 레지스트리 지문. 변경된 레지스트리의 오래된 색인은 버린다.

`tensor_codec.py`는 int64 노드, uint8 UTF-8 데이터, float64 값으로 변환한다.
int64 범위를 넘는 정수는 별도 타입 태그와 부호 있는 바이트로 정확하게 보존한다.
`.pt` payload는 텐서만 담고 `torch.load(weights_only=True)`로 읽는다.
파일을 임시 경로에 쓴 다음 교체한다. 기존 JSON 저장 동작도 유지한다.
의미 벡터를 역으로 풀어 정확한 문자열을 복원하는 방식은 아니다. 설명 문자열의
정확한 인코딩과 검색 벡터를 함께 유지하며, LLM 경계에서 JSON으로 재구성한다.

기능이 바뀌면 `TensorCatalog.refresh()`가 색인/계약을 갱신한다. 학습 성공 후
저장할 때도 적용한다. 이름으로 알려진 기능을 실행하는 런타임에는 LLM과
검색 인코더가 필요 없다. 자연어 조회에는 인코더가 별도로 필요하고 모델 가중치는
벡터 프로그램 파일에 포함하지 않는다.

## 요청 흐름

1. `search_goal`: 자연어 목표로 후보와 인자 계약을 텐서 파일에서 꺼낸다.
2. `prepare_N`: 후보 계약으로 자동 생성한 도구에 실제 인자를 넣는다.
   `width`, `x0`, `x1` 등을 필수로 받으며 숫자/경로/버퍼 형식을 구분한다.
   뺄셈 인자 역할은 이름이 아니라 알려진 셀 진리표와 scan 의미에서 추출한다.
3. 실제 요청 입력을 고정한 뒤 별도 caller 예제를 LLM에 제공한다.
   `verify_numeric` 또는 `verify_state`로 후보를 검사한다.
4. 예제를 만족하는 후보가 정확히 하나면 `execute_resolved`가 고정한 입력을
   한 번 실행한다. 여러 후보면 추가 근거를, 없으면 학습 예제를 요구한다.

검증은 복제한 레지스트리와 새 `MemoryHostContext`에서만 진행한다.
파일 목표는 변경하지 않을 파일까지 포함한 전체 스냅샷으로 비교한다.
후보 수·시간·실행 예산·예제 크기를 제한한다. 검색 유사도는 실행 허가가 아니다.
`--no-learning`은 학습 도구를 숨기고 호출도 거절한다. 단계에 없는 도구 호출,
검증 뒤 인자를 바꾸는 호출, 실제 실행 없는 성공 텍스트도 성공으로 처리하지 않는다.

구조화된 caller 예제가 있으면 그 입력 값에서 타입을 추출해 초기 후보도 좁힌다.
정답이나 평가용 기능 라벨에서 타입을 역추론하지 않는다. 예제가 없으면 초기
검색에는 타입 필터가 없다. 인자 확정 뒤 값에서 타입을 추출하고 호환 후보를
다시 검증한다. 구조화된 caller 예제의 입력/정답/스냅샷을 바꾸는 검증 호출은
거절한다. 자유 형식 텍스트는 이 동일성 검사가 적용되지 않는다. 검색 첫 후보의
이름으로 바로 작업을 실행하지 않는다. 필수 인자 계약도 LLM이 실제 숫자의 의미를
올바르게 읽었다는 보장은 아니다.

숫자 예제의 인자 개수도 실제 요청의 타입 계약과 같게 제한한다. 구조화된 caller
예제가 있고 실제 입력이 고정됐으면 먼저 검증 도구를 제공한다. 검증에 실패하면
질문할 수 있고, 이미 일치가 확인된 입력에는 실행 도구만 제공한다. 확보한 근거를
다시 확인해 달라는 불필요한 질문을 줄이기 위한 단계 제한이다.

## 실행 방법

기존 컨테이너의 Qwen과 MiniLM을 그대로 재사용한다. 검색용 선택 의존성은
`vectorpro[catalog]`이며 이미 설치된 환경에서는 추가 설치할 필요 없다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/catalog_agent/recheck --constrain-tools
```

HTTP 모델 서버를 사용할 때는 다음과 같다. `--endpoint`는 실제 서버 주소로,
`--model`은 그 서버의 모델 이름으로 바꾼다. 서버를 자동 생성하는 명령은 아니다.

```powershell
nerdctl exec vectorpro-test vectorpro-agent --program results/catalog_agent/program.pt --host-root results/catalog_agent/native --encoder /opt/vectorpro-models/multilingual-minilm --endpoint http://127.0.0.1:8000/v1/chat/completions --model model-name --no-learning --intent "Compute bitwise XOR of 12345 and 4567 at width 16" --evidence-file examples.json
```

`examples.json` 예시: `{"width":4,"operands":[[1,3],[5,3]],"targets":[2,6]}`.
프로그램 경로가 없으면 빈 런타임으로 시작하므로 기존 학습 파일을 먼저 복사한다.
실험 서버의 JSON 형식 제약은 외부 서버에 자동으로 적용되지 않는다.

## 실제 Qwen 검증 기록

같은 `vectorpro-test`의 Qwen3-0.6B Q8_0와 MiniLM을 재사용했다. 새 이미지,
컨테이너 또는 모델 다운로드는 없다. 모델은 실제 HTTP 도구 호출을 생성했다.
도구의 형식/단계는 제한하지만 인자 값과 정답을 스크립트 답변으로 대체하지 않는다.

| 기록 | 결과 | 의미 |
|---|---|---|
| `results/catalog_agent/final/summary.json` | 5/8 | XOR·뺄셈·곱셈 미지원 검증·모호한 요청 질문·한국어 XOR 성공 |
| `results/catalog_agent/phase_verified/summary.json` | 1/2 | 추가 단계 제한 후 실제 파일 XOR 성공, 자체 예제 생성 실패 |

첫 8건에서 파일 요청은 예제를 받고도 재확인을 요구해 실패했다. 후속 단계
제한으로 동일 파일 요청을 다시 실행해 반환값 4, 실제 바이트 `3532cab5`를 확인했다.
삭제 요청은 변경 없이 질문했지만 후보 검증을 수행하지 않아 엄격한 기준상 실패다.
자체 예제 생성은 같은 입력을 반복하고 잘못된 정답을 냈다. 중복 예제는 거절하고
실행하지 않았다. 두 실행 결과를 합쳐 최신 8건 성공률로 표현하지 않는다.
이 요청들은 수정 중 반복 사용한 진단 세트이며 독립 미사용 평가가 아니다.

`cli-smoke/result.json`에는 `.pt` CLI가 LLM/인코더 없이 32비트 XOR를 실행한
결과를 기록했다. 숫자 검증·실제 파일 작업·재저장과 학습 후 색인 갱신은 관련
테스트에서도 확인한다. 전체 테스트 기록은 [TESTING.md](TESTING.md) 참고.

## 검증의 경계

관련 테스트는 저장 호환, 재로딩 후 LLM 없는 실행, 자동 타입 추출, 동일 타입의
미지원 요청 거절, 모호한 예제, 전체 파일 비교, 실제 입출력 격리, 학습 후 색인 갱신,
실제 입력과 예제 입력의 분리 및 단계별 도구 제한을 확인한다.
실제 모델 실험의 원문과 실패도 `results/catalog_agent/`에 보존한다.

caller 정답으로 기능을 고르는 성공은 모델이 스스로 정답을 알아냈다는 뜻이 아니다.
모델이 예제를 잘못 만들면 알려진 기능도 거절할 수 있고, 잘못된 예제가 다른 기능을
가리키면 사람의 의도와 어긋날 수 있다. 숫자 입력의 해석 오류 역시 별도 문제다.
자동 정답 생성/독립적인 의도 확인, 기능이 늘어난 환경과 새 독립 요청 평가,
외부 모델 서버의 실제 호환성 및 macOS 검증은 남아 있다.
