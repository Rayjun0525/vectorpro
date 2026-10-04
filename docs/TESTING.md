# 테스트와 결과 확인

실제 Gemma의 호출·토큰·지연·성공률 비용 비교는 [AGENT_COST.md](AGENT_COST.md)를 참고한다.

리눅스 파일시스템 기반의 상태 학습·native 검증은
[LINUX_TARGET.md](LINUX_TARGET.md)의 재현 명령과 한계를 참고한다.

계약 초안의 학습·등록 검증은 [CONTRACT_LEARNING.md](CONTRACT_LEARNING.md)를 참고한다.

공통 계약의 저장 호환/직접 호출/CLI/어댑터 연결 검증과 재현은
[FUNCTION_CONTRACTS.md](FUNCTION_CONTRACTS.md)를 참고한다.

이전 Laya의 역할별 적합성 진단과 재현 명령은 [LAYA.md](LAYA.md)의
‘후속 사용 적합성 판정’ 절을 참고한다. 새 요청 24개/순서 반복/정답 계약 제공
인자 진단을 구분한다. Laya는 자동 요청 처리에 채택하지 않았다. 현재 기준은
Gemma이며 [GEMMA_BASELINE.md](GEMMA_BASELINE.md)를 참고한다.

작업 디렉터리는 Windows 호스트의 프로젝트 루트다. 명령은 PowerShell에서
실행하며 테스트 프로세스는 기존 `vectorpro-test` 안에서 실행한다.
컨테이너 내부 프로젝트 경로는 `/workspace`다.

## 환경 확인

```powershell
nerdctl ps -a --filter name=vectorpro-test
```

중지돼 있으면 `nerdctl start vectorpro-test`로 재사용한다. 컨테이너가 없으면
테스트를 멈추고 환경 손실을 알린다. 새 컨테이너·이미지를 만들거나 컨테이너를
삭제하지 않는다. 현재 컨테이너에는 PyTorch, llama-cpp-python 0.3.36,
transformers 4.57.6, laya 0.3.26 및 아래 모델이 설치돼 있다.

| 자원 | 컨테이너 내부 위치 |
|---|---|
| 실제 소형 LLM | `/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf` |
| 다국어 검색 인코더 | `/opt/vectorpro-models/multilingual-minilm/` |
| 실험의 원본 프로그램 | `/workspace/results/initial_model/program.json` |

모델은 저장소에 포함하지 않는다. MiniLM 파일만 누락됐으면
`nerdctl exec vectorpro-test python experiments/setup_retrieval_model.py`로 같은
컨테이너에 내려받는다. 기존 download.json이 있으면 기록된 revision을 재사용한다.
Gemma 재설치는 `nerdctl exec vectorpro-test python experiments/setup_gemma_model.py`로
고정 revision과 SHA-256을 확인한다. Laya/Qwen 가중치는 제거했으며 과거 결과는
유지한다. Laya 설치기는 Gemma를 제거하는 이전 교체 작업이므로 현재 실행하지 않는다.
로컬 runner의 기본 모델은 Gemma이며 추론 시 자동 다운로드하지 않는다.
원본 프로그램 재생성 방법은 README의 initial-model 절에 있다.

## 변경별 검증

경로 검증은 원문 선택지, 한글/공백/여러 점, 파일명 숫자 제외, 경로 누락 질문,
검증 예제 경로 제외, 요청에 없는 실제 존재 파일도 접근 전에 거절하는지 확인한다.
새 평가 파일 `experiments/requests/gemma_paths_first_use.json`의 기대 경로/스냅샷은
평가기에만 전달한다. 개발 재검증은 `results/gemma_paths/development`, 기존 자동
학습 요청의 진단은 `teacher-diagnostic`, 새 경로 요청은 `first-use`로 구분한다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/gemma_paths/reproduction --constrain-tools --evaluation experiments/requests/gemma_paths_first_use.json --max-calls 6
```

기존 고정 요청의 재실행은 replay이며 최초 미사용 평가일 때만 `--evaluation-kind
first-use`를 명시한다. 옵션 자체가 독립 평가임을 증명하지는 않는다.

최신 전체 회귀 186 passed, 280.11초, 관련 32 passed, 9.90초.
새 경로 요청의 전체 성공은 4/6이며 실행 경로만 일치한 것은 5/5다. 실패 2건은
값/폭을 바꿔 실제 테스트 대상 바이트가 잘못 변경됐다. 보존 파일 유지도 확인했다.
이 결과로 튜닝하지 않았다. NAND 진단에서는 학습/별도 평가가 진행되지 않았다.
실험별 의미와 코드/데이터 SHA는 `results/gemma_paths/verification.json` 참고.

최신 후속 입력/완료 변경 전체 회귀는 **183 passed, 338.79초**, 관련 테스트는
29 passed, 17.28초다. 실제 모델 추론과 병행했으므로 이전 시간과 직접 비교하지 않는다.
모호한 caller 근거의 자동 승인 방지, 마지막 허용 턴의 정상 종료, 실제 인자/근거
분리, 성공 후 모델 추가 호출 없음 및 숫자 선택지에 출력값을 넣지 않음을 확인한다.
새 첫 평가 7/8은 실제 폭/인자/완료 상태를 포함한다. 한국어 파일 경로 추출 실패
1건은 원본 불변을 확인했고 결과를 보고 다시 튜닝하지 않았다. 원문은
`results/gemma_binding/first-use`, 소스/데이터 SHA와 한계는 `verification.json`에 있다.
최초 실행에는 `--evaluation-kind first-use`를 명시했고 위 재현에는 기본 replay를
사용한다. 구분 옵션이 새로운 미사용 평가임을 자동으로 증명해 주지는 않는다.

후속 입력 준비/자동 caller 검증/완료 반환은 새 8건의 폭·입력·완료 상태까지 검사한다.
기대 출력/인자는 평가기에만 전달하며 모델에는 원래 요청과 명시적 caller 근거만
사용한다. 재현은 새 결과 경로를 지정한다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/gemma_binding/reproduction --constrain-tools --evaluation experiments/requests/gemma_binding_first_use.json --max-calls 6
```

이미 사용한 평가 요청을 재실행하면 새 독립 평가가 아니다. 기존 실패 재사용 개발
기록은 `results/gemma_binding/development`, `literal-development`로 분리한다.

Gemma 교체 후 전체 회귀는 **181 passed, 374.36초**다. 관련 어댑터/프로토콜
범위는 27 passed, 10.14초다. 실제 모델 테스트와 병행한 실행 시간이므로
이전 시간과 직접 성능 비교하지 않는다. 테스트는 모두 같은 컨테이너에서 실행했다.
새 테스트는 Gemma 역할 변환에서 실제 입력/검증 근거/도구 인자가 유지되고
평문을 가짜 도구 호출로 바꾸지 않는지 확인한다.
실제 Gemma 재현은 7/12이며 기존 출력/파일 기준이다. 잘못된 폭 해석과 실행 후
추가 질문은 별도 한계로 기록했다. 실패한 파일 변환 2건의 원본 불변, NAND 등록
없음도 확인했다. 기록은 `results/gemma_model/verification.json`, 원문은
`results/gemma_model/catalog-v2/`다. 기존 고정 요청의 재현이며 독립 정확도가 아니다.

최종 근거 선검사/작은 예제 문맥/파일 실행 경계 변경은 전체
**179 passed, 247.57초**, 관련 범위 27 passed, 8.85초다.
실행 경계 수정 전에는 전체 178건(241.22초), 관련 26건(8.94초)이었다.
새 요청 첫 평가 9/12와 같은 실패 사례의 수정 후 Qwen 재검증 1/1을 구분한다.
후자는 질문 반환·실제 파일 불변·학습 없음 확인이며 독립 정확도 평가가 아니다.
자세한 기록은 `results/catalog_agent/v2-verification.json`에 보존한다.
아래 173건 기록은 이전 구현 시점이다.

2026-10-04 최신 텐서 저장/카탈로그 연결 변경을 포함한 전체 회귀는
**173 passed, 232.15초**다. 기존 `vectorpro-test`만 사용했다. 관련 범위를 먼저
실행한 결과는 21 passed, 15.61초이며 이를 전체 결과에 더하지 않는다.
실행 명령/모델 기록은 `results/catalog_agent/verification.json`에 보존한다.

| 변경 범위 | 우선 실행 | 확인할 내용 |
|---|---|---|
| LLM 도구 입력/오류 처리 | `tests/test_agent.py` | 학습·실행·저장, 입력 검증, 파일 비변경 |
| 의미 검색/타입 필터 | `tests/test_catalog_retrieval.py` | 잘못된 타입 제외, 후보 없음, 정답 라벨과 검색 독립 |
| 정식 텐서 저장·카탈로그 어댑터 | `tests/test_semantic_catalog.py` | 저장/재로딩, 예제 격리, 동일 타입 미지원 요청, 인자 고정, 예제 변조 차단 |
| 텐서 저장 실험 | `experiments/tensor_catalog.py` | 정확 복원, 수정 재저장, 실제 숫자/파일 실행 |
| 검색 품질 | `experiments/catalog_retrieval.py` | 의미만/타입 포함 비교와 별도 평가 요청 |
| 실제 모델의 도구 사용 | `experiments/local_small_llm.py` | 원문 모델 출력, 도구 결과, 기능 등록/파일 결과 |
| 학습기·실행 커널·정식 저장 형식 | 전체 pytest | 기존 기능의 회귀 여부 |

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest tests/test_agent.py tests/test_catalog_retrieval.py -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest tests/test_semantic_catalog.py tests/test_agent.py tests/test_catalog_retrieval.py -q
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

앞선 전체 회귀 기록은 당시 157건 통과(291.39초)다. 이후 추가한 검색 테스트
2건은 별도 실행해 통과했고, 최신 어댑터 테스트 6건도 별도 통과했다. 이 숫자를
현재 트리 전체를 다시 실행한 결과로 합산하지 않는다. 문서 변경만 한 경우에는
링크·명령·기록의 일관성을 확인하며 전체 테스트를 반복하지 않는다.

main 반영 전 최종 확인에서 위 두 테스트 파일을 함께 실행해 8건이
통과했다(6.81초). 이번 커밋 시점의 관련 범위 검증 결과다.

## 텐서 파일과 검색 실험 재현

기존 결과를 보존하려면 아래처럼 새로운 결과 하위 디렉터리를 지정한다.
프로그램 실행으로 이 디렉터리에 input.bin 및 program.pt가 생성/변경된다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/catalog_retrieval.py --root results/catalog_retrieval/recheck
```

각 디렉터리의 summary.json을 확인한다. 예외 없이 프로세스가 끝났다는 것만으로
의미 검색이 정확하다고 판단하지 않는다. 검색 오류도 정상적으로 결과에 기록된다.

`tensor_catalog.py`의 원래 Qwen 풀링 실험은 보존된 기록이다. 현재 기본 검증에는
포함하지 않으며 별도 호환 GGUF를 `--model`로 명시해야 실행한다. 실제 검색은
MiniLM을 사용하고 Gemma 교체로 검색 색인/벡터 프로그램을 다시 학습하지 않는다.

- 텐서 복원 기준: exact_roundtrip, unicode_types_roundtrip, native_file_passed,
  metadata_edit_roundtrip이 true, numeric_heldout이 100.
- 검색 기준: results의 각 평가 세트별 top1/top3/cases와 search_then_execution 확인.
  원본 데이터·색인·계약 복원은 exact_roundtrip과 catalog_contracts_roundtrip 확인.
- 최종 기록: 원래 요청 4/4 및 실제 실행 4/4. 새로운 영어/한국어 24건은 의미만
  top1 12/24·top3 21/24, 명시적 입력 타입 포함 top1 24/24.
- 타입 포함 결과는 요청자가 정확한 타입을 전달한 조건이다. 16건은 타입으로
  후보 하나가 남고 8건은 복수 후보에서 의미 검색으로 구분했다.
- evaluation_protocol 확인: development는 기존 4건, heldout은 개선 중 확인한
  진단 24건, final_holdout은 마지막 비교 전에 따로 고정한 새 24건이다.
  이미 확인한 요청으로 수정한 뒤에는 새 독립 평가 세트를 추가해야 한다.

## 실제 소형 LLM 재현

후속 평가 명령은 아래와 같다. 기존 결과 경로가 있으면 거절하므로 새 경로를
사용한다. 고정된 요청의 재실행은 재현이며 새 독립 평가로 취급하지 않는다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/catalog_agent/v2-reproduction --constrain-tools --evaluation experiments/requests/catalog_agent_holdout_v2.json --max-calls 12
```

`checks`의 파일 전체 비교, 실제 출력, 미지원 근거 검사, 네이티브 효과/학습 없음,
등록·재로딩·별도 평가를 확인한다. evaluation_sha256과 protocol을 기록한다.
caller 예제 요청과 자체 예제/학습 요청을 분리해서 해석한다. 사례 생성 스크립트의
Python 정답 계산은 평가용이며 자체 예제/학습 요청에는 전달하지 않는다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --root results/local_small_llm/recheck --constrain-tools
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/catalog_agent/recheck --constrain-tools
```

모델이 실제 추론하고 로컬 HTTP를 통해 기존 어댑터에 도구 호출을 전달한다.
도구 선택/인자를 스크립트 답변으로 대체하지 않는다. --constrain-tools는 이
실험 서버에서 출력 형식을 제한하며 외부 서버에 자동 적용되는 설정은 아니다.
종료 시 임시 HTTP 서버를 닫고 모델/컨테이너는 재사용할 수 있게 남긴다.
현재 출력은 모델 턴당 기본 512토큰으로 제한한다. `--max-tokens`로 바꿀 수 있다.

카탈로그 경로는 caller 예제를 실제 인자 확정 뒤 별도 메시지로 전달한다.
모델이 예제를 스스로 만든 `catalog_xor_self`는 별도로 구분한다. 파일 성공은
반환값뿐 아니라 실제 파일 바이트까지 확인한다. 미지원 연산은 검증 결과가
needs_learning_examples이고, 질문을 반환하며 실행/학습/파일 변경이 없어야 통과다.
같은 8개 요청으로 개선했으므로 독립 평가나 범용 성공률로 표현하지 않는다.
원문 모델 출력과 도구 결과는 각 시나리오의 transcript.json/result.json에 있다.
카탈로그 진단 `final`은 5/8이며, 후속 `phase_verified`는 1/2다. 후자는 파일
흐름 보완 뒤 실제 파일 성공과 자체 예제 생성 실패를 분리한 기록이다. 둘을 합쳐
최신 8건의 정확도로 취급하지 않는다. 상세 실패는 CATALOG_AGENT.md에 기록했다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/local_small_llm.py --catalog results/catalog_retrieval/final/program.pt --root results/catalog_agent/phase-recheck --constrain-tools --scenarios catalog_file catalog_xor_self
```

summary.json, 각 시나리오의 result.json/transcript.json을 함께 확인한다.
기존 형식 제약 비교는 연산·질문·파일 3/4 성공, 새 NAND 학습은 실패했다.
최소 사례 수 제한 이후 NAND 재검증에서는 중복 사례를 거절하고 질문을 반환했다.
따라서 자동 학습의 정답 생성 품질이 해결됐다고 표현하지 않는다.

플랫폼 검증은 [PORTABILITY.md](PORTABILITY.md), 상세 설계/결과는
[TENSOR_CATALOG.md](TENSOR_CATALOG.md)와 [LLM_ADAPTER.md](LLM_ADAPTER.md)를 참고한다.
macOS 검증은 사용자 결정으로 보류한다.
