# 텐서 카탈로그 번외 검증

2026-10-04 사용자 요청으로 정확히 복원되는 정보와 의미 검색 벡터를 함께
저장하는 작은 프로토타입을 검증했다. 아래는 당시 기록이다. 이후 정식 `.pt`
런타임 저장과 LLM 연결을 추가했으며 [CATALOG_AGENT.md](CATALOG_AGENT.md)에 설명한다.

## 저장 구조

`experiments/tensor_catalog.py`는 기존 런타임 파일의 전체 데이터를 정수 타입
태그/트리 구조를 담은 int64 텐서, UTF-8 문자열의 uint8 텐서, float64 값 텐서로
변환한다. 메타정보와 실행 데이터 모두 이 트리에 들어간다. 단순히 JSON 문자열
전체를 바이트로 감싼 방식은 아니다. 값이 문자열인 설명은 여전히 언어로
작성된 내용을 정확하게 인코딩한 것이며 언어 의존성이 사라지는 것은 아니다.

같은 `program.pt`에는 실제 Qwen3-0.6B Q8_0 모델에서 추출한 기능 이름/설명
평균 풀링 벡터도 저장한다. 무작위 실행 주소 벡터를 의미 임베딩이라고 부르지
않는다. 이 모델은 문장 검색용으로 훈련된 임베딩 모델이 아니므로 검색 품질은
별도로 측정한다. 파일의 payload는 텐서만 담은 사전이고 weights_only=True로
읽는다. PyTorch 컨테이너 헤더/텐서 이름까지 벡터화했다는 뜻은 아니다.

## 결과

- 12개 기능의 전체 데이터 저장·복원 일치.
- 한글·null·불리언·정수·실수·빈 객체·빈 배열 복원 일치.
- 복원한 실행 구조로 XOR 미사용 16비트 입력 100개 일치.
- 복원한 파일 map이 실제 파일 `0007ff80`을 `3532cab5`로 변환하고 4 반환.
- 메타정보를 JSON 객체로 꺼낸 후 태그 수정·재인코딩·같은 파일 저장·재로딩 일치.
  해당 검색 벡터도 갱신하고 실행 provenance는 변경되지 않았음을 확인.
- 의미 검색 영어 요청 4건: top1 1/4, top3 3/4. 파일 map은 top3에도 없었다.
- 원본 JSON 59,087바이트, 텐서 파일 290,957바이트. 이 표현은 크기 최적화가 아니다.

결과: `results/tensor_catalog/summary.json`. 텐서 파일: `program.pt`.
LLM 교환 형식으로 복원한 목록: `exported-catalog.json`.
이번 실험은 JSON 목록을 복원하는 경계까지 확인하며 LLM에 이 목록을 넣고
전체 요청 루프를 다시 돌린 테스트는 아니다. 숫자/파일 실행은 이름으로 선택해
검증했으며 검색 top1을 신뢰해 실행한 것이 아니다. 의미 벡터는 실행 규격이나
검증 근거를 대체하지 않는다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/tensor_catalog.py
```

기존 vectorpro-test 및 이미 설치된 모델만 재사용했다. 모델/의존성 추가 없음.
남은 과제는 검색용 모델과 독립 평가 확대, 타입/효과 기반 후보 검증, 저장 버전과
기능 ID 규약, 선택적인 실제 런타임 저장/LLM 연동이다. macOS 검증은 보류 상태다.

## 검색 정확도 개선 검증

후속 요청으로 `experiments/catalog_retrieval.py`를 추가했다. Qwen 평균 대신
문장 유사도용으로 훈련된 [multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)을 사용한다.
117,653,760 파라미터, 384차원, 마스크 평균 풀링과 L2 정규화다.
revision은 `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`다.
모델/토크나이저 총 479,724,655바이트를 같은 컨테이너의
`/opt/vectorpro-models/multilingual-minilm`에 저장했다. transformers 4.57.6을
같은 컨테이너에 추가했고 새 이미지/컨테이너는 생성하지 않았다.

동일한 “learned state transformation” 설명 대신 호출·라우팅·분기 텐서에서
검색 문서를 추출한다. 범용 host 연산/알려진 셀 진리표의 해석 규칙은 사람이
제공한 기계 의미이며 새 작업 로직을 학습한 것이 아니다. 파일 바이트를 직접
입력 값으로 바꾸는지, 계산 결과로 바꾸는지, 조건/복수 경로가 있는지를 구분한다.
모든 기능 이름을 opaque 이름으로 바꿔도 동일한 문서가 나오는 검증도 수행했다.

검색 후 caller가 명시적으로 전달한 input_types와 일치하지 않는 후보를 제외한다.
host 타입은 HOST_TYPES, 배운 절차 타입은 provenance에서 얻는다.
평가 정답으로 타입을 역추론하지 않으며 후보가 없으면 빈 목록을 반환한다.

| 평가 | 첫 후보 성공 | 상위 3개 포함 |
|---|---:|---:|
| 기존 Qwen 평균, 기존 요청 | 1/4 | 3/4 |
| MiniLM + 텐서 문서, 기존 요청 | 4/4 | 4/4 |
| MiniLM + 텐서 문서, 최종 새 요청 | 12/24 | 21/24 |
| 위 방식 + 명시적 입력 타입, 최종 새 요청 | 24/24 | 24/24 |

최종 새 요청은 영어 12건·한국어 12건이다. 타입으로 후보 하나가 남는 16건과
후보가 여러 개 남아 의미 검색이 필요한 8건을 포함하며, 후자의 8건도 통과했다.
범용 100% 정확도라는 뜻은 아니다. 처음 추가한 24건은 개선 도중 결과를 확인해
진단용으로 취급하고, 최종 24건은 이후 따로 고정했다. 모델을 추가 훈련하지 않았다.

색인·정확한 원본 데이터·문서·타입 계약·모델 revision은 같은 program.pt에
저장/복원했다. 기존 요청 4건에서 실제 검색 첫 후보를 실행해 XOR·뺄셈·파일
XOR·파일 fill이 모두 맞는지 확인했다. 검색 단위 테스트 2개도 통과(1.93초)했다.
기존 핵심 실행기/학습기/저장 API는 변경하지 않았다.

```powershell
nerdctl exec vectorpro-test python experiments/setup_retrieval_model.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/catalog_retrieval.py --root results/catalog_retrieval/final
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest tests/test_catalog_retrieval.py -q
```

결과는 `results/catalog_retrieval/final/summary.json`, 평가 요청은
`experiments/requests/retrieval_evaluation.json`이다. 앞선 단계 결과도 보존했다.
이후 LLM/요청 어댑터의 타입 추출과 조회 연결, 예제 기반 미지원 요청 거절을
구현했다. 현재 결과와 남은 한계는 [CATALOG_AGENT.md](CATALOG_AGENT.md) 참고.
