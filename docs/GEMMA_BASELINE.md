# Gemma 기준과 최적화 범위

사용자 요청으로 Laya 적합성 시험을 마치고 Gemma 3 1B Instruct Q8_0을 복원했다.
같은 `vectorpro-test` 컨테이너와 `python:3.12-slim` 이미지를 재사용한다.
새 이미지/컨테이너 생성 없이 Laya의 정확히 확인된 모델 디렉터리만 제거했다.
MiniLM 검색 인코더와 배운 텐서 프로그램, 과거 실험은 유지한다.
Laya 제거 기록은 `results/gemma_return/laya-retirement.json`이다.

## 모델 기준

- Repo: `ggml-org/gemma-3-1b-it-GGUF`
- Revision: `f9c28bcd85737ffc5aef028638d3341d49869c27`
- File: `/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf`
- Bytes: 1,069,306,368
- SHA-256: `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- Native Gemma template, JSON-schema tool constraints, CPU llama-cpp-python 0.3.36.

이 모델은 Laya보다 약 391MB를 더 사용한다. 컨테이너 내부 파일을 제거해도
Windows 호스트의 가상 디스크가 즉시 같은 만큼 축소되는 것은 아니다.

## 명시된 폭과 숫자 인자 구분

`binding_tools`는 원문 경로를 제외한 문자열에서 `width 16`, `폭은 16`,
`16-bit register`, `16비트 레지스터`와 같은 명시 표기를 읽는다. 서로 다른
폭이 없고 1..32 범위의 한 값이면 width enum을 그 값 하나로 제한한다.
다른 숫자 인자의 선택지는 그대로 유지한다. 기능별 알고리즘이나 결과를
계산하지 않는다. 작업 전용 분기는 실행기에 추가하지 않았다.

폭 표기가 없거나 여러 값이 충돌하면 기존 숫자 선택지를 유지한다. 모든 자연어
폭 표현을 이해하는 문법이 아니며 모호성을 자동 해소하지 않는다.
`16비트로 실행`처럼 현재 좁은 문법 밖의 표현, 마스크/값/플래그의 역할,
여러 경로의 대상/보존 역할은 여전히 모델 해석의 한계다.
모델이 enum을 무시해도 기존 백엔드 검사가 거절한다. 이 검사는 요청 의도 전체나
일반적인 실행 정확성에 대한 증명이 아니다.

## 재현

```powershell
nerdctl ps -a --filter name=vectorpro-test
nerdctl exec vectorpro-test python experiments/setup_gemma_model.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_binding_width.py tests/test_semantic_catalog.py tests/test_local_model_protocol.py
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/gemma_return/path-reproduction --constrain-tools --evaluation experiments/requests/gemma_paths_first_use.json --max-calls 12
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/gemma_return/width-reproduction --constrain-tools --evaluation experiments/requests/gemma_width_first_use.json --max-calls 12
```

결과 디렉터리는 새 이름을 사용한다. 최초 width 평가 자료는 추론 전에
`experiments/create_gemma_width_cases.py`로 고정했다. 이 생성기는 scoring용
파일 기대값만 계산하며 모델에 전달하지 않는다. caller 증거는 명시적으로
제공한다. 재현 명령은 replay이며 이미 실행한 요청을 첫 평가라고 표시하지 않는다.
파일 실행은 결과 디렉터리 아래 전용 테스트 파일에만 수행한다.

## 현재 단계와 남은 과제

이번 관련 테스트는 32 passed, 7.99초이며 전체 회귀는 192 passed, 362.21초다.
전체 회귀와 실제 추론을 병행한 시간으로 속도 비교에 쓰지 않는다.
이전 경로 6건 replay는 5/6으로, ‘폭은 16’ 오류는 개선했으나 ‘16비트로 실행’
표기의 bare_korean_fill은 폭31/값16으로 대상 파일을 잘못 썼다. 보존 파일은 유지됐다.
같은 개발 요청 재검증이므로 새 독립 정확도 개선으로 표현하지 않는다.

추론 전에 별도로 고정한 명시적 폭 요청 6건은 **6/6** 통과했다.
폭 16/24/32와 값, 경로, 실행 완료, 반환값, 전체 파일 스냅샷을 함께 비교했다.
4개 fill과 2개 XOR 변환으로 범위가 좁으며 caller 증거를 제공한 조건이다.
결과 이후 코드/요청/프롬프트 튜닝은 없다. 자동 학습이나 범용 의도 정확성을
증명하지 않는다. 원문은 `results/gemma_return/width-first-use` 및 `path-replay`,
모델/코드/데이터 SHA와 한계는 `results/gemma_return/verification.json`에 있다.

초기 학습·실행 기반과 Gemma 도구 연결은 있다. 현재는 정확한 입력 전달을
개선하는 단계다. 공통 기능 계약과 모델 없는 직접 호출 API의 통합,
숫자/경로 역할·입력 타입 검증, 독립적인 정답 근거와 자동 학습 준비,
더 넓은 작업 표현과 별도 일반화 검증이 남았다. 알려진 기능 실행은 LLM 없이
가능하다. macOS는 사용자 요청으로 보류한다. 전반적인 완성률 숫자는 정하지 않았다.
