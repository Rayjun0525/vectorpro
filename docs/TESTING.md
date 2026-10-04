# 테스트와 결과 확인

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
transformers 4.57.6 및 두 모델이 이미 설치돼 있다.

| 자원 | 컨테이너 내부 위치 |
|---|---|
| 실제 소형 LLM | `/opt/vectorpro-models/Qwen3-0.6B-Q8_0.gguf` |
| 다국어 검색 인코더 | `/opt/vectorpro-models/multilingual-minilm/` |
| 실험의 원본 프로그램 | `/workspace/results/initial_model/program.json` |

모델은 저장소에 포함하지 않는다. MiniLM 파일만 누락됐으면
`nerdctl exec vectorpro-test python experiments/setup_retrieval_model.py`로 같은
컨테이너에 내려받는다. 기존 download.json이 있으면 기록된 revision을 재사용한다.
Qwen GGUF가 누락된 경우에는 [LLM 검증 기록](LLM_ADAPTER.md)의 모델/revision을
확인하고 복구한다. 원본 프로그램 재생성 방법은 README의 initial-model 절에 있다.

## 변경별 검증

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
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/tensor_catalog.py --root results/tensor_catalog/recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python experiments/catalog_retrieval.py --root results/catalog_retrieval/recheck
```

각 디렉터리의 summary.json을 확인한다. 예외 없이 프로세스가 끝났다는 것만으로
의미 검색이 정확하다고 판단하지 않는다. 검색 오류도 정상적으로 결과에 기록된다.

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
