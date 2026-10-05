# 어시스턴트 해석으로 교체한 통제 시험

2026-10-05. 사용자가 ‘네가 인터셉트해서 모델 파워 때문인지 시험해 달라’고 요청했다.
작은 모델의 해석 결과 대신 어시스턴트가 직접 작성한 조건을 넣어 같은 실행 경로가
동작하는지 확인했다. 이 시험은 앞단과 실행기를 분리하는 진단이며, 모델 크기만의
효과를 증명하는 독립 모델 비교는 아니다.

## 절차와 정답 노출

직전 대비 학습의16개 요청을 재사용했다. 어시스턴트는 이미 평가 라벨을 읽은
상태에서 원문을 해석해 experiments/assistant_intercept_answers.json에 응답을
고정했다. 각 응답은 supported/unsupported/contradictory/unclear와 두 인자 최종
상태, 일부 명시 경로와 이유다. 시험 코드가 기대 정답을 읽어 응답을 생성하는
oracle는 아니지만, **라벨을 본 어시스턴트의 재시험**이므로 독립 정확도라고 하지 않는다.
실시간 외부 고성능 모델 API의 성능이나 비용을 측정한 결과도 아니다.

같은 요청을 설치된 Gemma3 1B Q8_0에 실제로 보내 두 출력 형식을 비교했다.

- 기존 rules: 상태 조건과 전체 변경 제한을 직접 작성한다.
- compact: 어시스턴트와 같은 결정/두 인자 상태 형식으로 답한다. 백엔드가
  허용 변경 목록을 기계적으로 구성하므로 복잡한 규칙 배열을 만들 부담을 줄인다.

두 방식 모두 의미적으로 부정확한 응답은 채택하지 않으며 Gemma에게 사용자 파일,
평가 정답, 실행/학습/승인 도구를 제공하지 않는다. compact도 기본 경로에 넣지 않았다.
고정 응답 변환은 일반적인 조건 형식 변환이며 작업 이름별 Python 절차가 아니다.

## 같은16건의 목표 해석 결과

| 해석 경로 | 전체 맞음 | 지원 목표 맞음 | 오답 초안 |
|---|---:|---:|---:|
| 어시스턴트의 고정 응답 재시험 | 16/16 | 12/12 | 0 |
| 실제 Gemma 기존 rules | 2/16 | 0/12 | 0 |
| 실제 Gemma compact | 4/16 | 0/12 | 1 |

rules는10건을 needs_input,6건을 goal_draft_failed로 반환했다. 목표를 처리하지
않아 오답 초안은0이어도 지원 기능을 성공한 것은 아니다. compact는15건을
needs_input,1건을 잘못된 needs_goal_review로 반환했다. 미지원/모순/모호4건은
compact에서 모두 보류했지만 지원12건의 성공은 없었다.

실제 calls/seconds/total_tokens: rules16/140.37/12,754,
compact16/97.11/7,093. 시간은 모델 추론 합이며 전체 에이전트 비용이나 전용
속도 비교가 아니다. 어시스턴트의 고정 답 파일을 읽는 비용과 이 숫자를 비교하지 않는다.
원본 메시지/도구/응답/시간/토큰은 각 raw_calls.json에 보존했다.

기존 rules 도구 스키마는 unchanged_except 목록의 최소 길이가1이어서, 전체
보존의 빈 변경 목록을 표현하는 데 한계도 있다. 이 점은 모델만의 실패로 돌리지
않는다. compact에서는 빈 변경 목록을 백엔드가 만들 수 있지만 전체 지원 요청
처리로 이어지지 않았다. 한 모델/두 프롬프트/작은 재시험이며 크기만의 인과 비교가 아니다.

## 저장된 텐서의 실제 리눅스 실행

기존 results/goal_router_execution_replay/program.pt에는 학습된 복사/이동 계약이
있다. 어시스턴트가 제공한 목표를 대상으로 각 계약의 정방향/역방향 인자 연결을
새 MemoryHostContext 두 입력에서 시험해, 목표 전체 파일/빈 폴더 조건을 만족하는
경로가 하나일 때만 채택했다. 기대 평가 라벨은 이 경로 선택에 쓰지 않는다.

그 다음 별도 native 루트에 빈 바이트, 바이너리, 새 한국어 문자열의 세 입력을
만들어 해당 저장 텐서 계약을 실행했다. 마지막 채점에서만 원래 기대 조건과
전체 파일/디렉터리 결과를 비교했다. 특정 작업의 순서를 Python에 구현한 것이
아니며 실제 읽기/쓰기/이동은 저장된 텐서 계약의 명령이 수행했다.

복사/이동 요청10개 × 입력3개 = **30/30 정확히 실행**했다. 다른 파일/빈 폴더도
보존했다. 전체 보존/대상 삭제2개는 이 시험 프로그램에 해당 계약이 없어 실행하지
않고 no_unique_stored_contract를 기록했다. 미지원/모순/모호4개도 실행하지 않았다.
이 두 계약의 부재는 전체 커널이 그 작업을 학습할 수 없다는 증거가 아니다.

role-only 요청에는 호출자 시험 문맥이 source.bin/target.bin의 역할-경로 연결을
제공했다. 두 파일명이 명시된 요청은 고정 응답에 input-x.bin/archive-y.bin을
명시했다. 따라서 임의 자연어 경로 추출 정확도까지 증명한 것은 아니다.
기본 프로그램 파일의 바이트가 변하지 않은 것도 확인했다.

compact 대조 시험에서도 같은 native30개를 재실행했고30/30이었다. 동일 사례
재시험이므로60개의 독립 검증으로 합산하지 않는다. Gemma의 틀린 초안으로
native 실행을 시험하지 않았다.

## 결론과 남은 것

정확한 목표/인자 연결을 주면 저장된 텐서가 수행할 수 있다는 근거를 강화했다.
이번 실패의 큰 병목은 **자연어 해석·목표 작성 앞단**이며 실행기가 모두 실패하는
상태는 아니다. 작은 모델의 출력 능력/프롬프트/도구 형식과 현재 벡터 표현·학습
데이터가 함께 영향을 줄 수 있다. ‘더 큰 모델 하나면 모두 해결’이라고 단정하지 않는다.

다음은 정답을 보지 않은 강한 모델의 동일 인터페이스 비교, 그 해석의 독립 상태
검증, 검토된 대비 데이터 축적, 미지원 효과와 역할/경로 연결의 정확도를 분리해서
평가하는 것이다. 이 고정 응답16개를 자동으로 교사 데이터/등록 계약에 넣지 않았다.

## 검증과 재현

관련37 passed (6.77초). 역방향 조건/전체 보존 검사, 미지원 응답의 실행 목표
부재, 불완전한 응답 거부와 compact 경계를 포함한다. 런타임/학습기/저장 형식은
변경하지 않아 전체 회귀398 passed (282.65초)는 직전 기록이다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.assistant_intercept --live-gemma --output results/assistant_intercept_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.assistant_intercept --live-gemma --compact-gemma --output results/assistant_intercept_compact_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q tests/test_assistant_intercept.py tests/test_goal_evidence.py tests/test_goal_draft.py tests/test_verified_reuse.py
```

결과 폴더는 기존 기록을 덮어쓰지 않는 새 이름을 사용한다. 기존 vectorpro-test
하나/기존 이미지/기존 설치 모델을 재사용했다. 모델 다운로드/이미지 빌드/새
컨테이너 생성은 없다. 원본은 results/assistant_intercept와
results/assistant_intercept_compact의 answers/cases/summary/raw_calls.json이다.
