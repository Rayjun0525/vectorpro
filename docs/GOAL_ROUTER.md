# 목표 분류 학습과 보수적 합의 검증

2026-10-05. 이번 변경은 자연어에서 목표 초안을 만드는 앞단의 정확도 개선이다.
리눅스 실행 커널이나 작업 전용 알고리즘을 새로 작성한 것이 아니다. 아직 임의의
자연어 요청을 프로그래밍 언어 수준으로 정확하게 해석한다고 주장하지 않는다.

## 방법

기존 MiniLM 문장 벡터는 ‘복사하고 원본 보존’과 ‘이동하고 원본 삭제’를 높은
유사도로 혼동했다. 기존 예제13개에 보존/삭제, 이동 방향, 미지원 효과를 대비한
실험 작성자 라벨30개를 추가했다. 총43개는 `experiments/goal_router_training.json`에
고정했다. 실제 사용자가 확인한 이력이나 자동으로 발견한 정답은 아니다.

고정 인코더의 정규화된 벡터 X와 목표별 one-hot 라벨 Y로 작은 선형 분류기를
학습한다. 가중치는 W = Xᵀ(XXᵀ + λI)⁻¹Y, λ=0.05이다. 조회는 xW를 계산하며
최고 점수0.6 이상, 차순위와의 차이0.2 이상일 때만 목표 후보를 사용한다.
이 점수는 확률이 아니다. MiniLM이나 Gemma의 가중치는 변경하지 않았다.

`GoalMemory(..., routing="ridge")`는 이 분류기를 사용한다.
`routing="ridge_consensus"`는 기존 최근접 검색도 유사도0.55/차이0.03을 통과하고
같은 목표를 골라야 초안을 반환한다. 두 판단은 같은 인코더/예제를 공유하므로
통계적으로 독립된 증거가 아니다. 의견 일치도 의미의 정답을 증명하지 않는다.
불일치와 약한 점수는 `needs_input`, 일치는 여전히 `needs_goal_review`이다.
기존 파일과 기본 `nearest` 동작을 보존했고 새 모드는 명시적으로 선택한다.

분류 가중치·목표 그룹·예제·벡터·임계값은 같은 프로그램 파일에 저장한다.
별도 작업 메타파일을 런타임에서 요구하지 않는다. 로드할 때 다시 학습하지 않으며
차원·유한값·그룹 대응·임계값·인코더 신원을 확인한다. 기존 version 1 저장 형식의
선택적 router 필드이며 `.pt`/`.json` 모두 검증했다.

## 평가와 한계

첫 스트레스 검사24건은 학습 예제와 다른 보존/삭제/반대 방향/모순/미지원 문장이다.
결과를 본 뒤 합의 조건을 추가했으므로 합의 방식의 독립 평가로 사용하지 않았다.
`results/goal_router_stress`에 당시 결과를 보존했다.

| 방식 | 전체 맞음 | 지원 목표 맞음 | 잘못된 초안 |
|---|---:|---:|---:|
| 최근접, 예제13개 | 9/24 | 6/18 | 8 |
| 최근접, 예제43개 | 12/24 | 10/18 | 7 |
| 선형 분류, 예제43개 | 13/24 | 7/18 | 1 |

선형 분류의 오답은 ‘source와 destination 모두 현재 상태 유지’에 역방향 복사를
제안한 것이었다. 점수0.84/차이0.49로 높았으므로 점수만으로 안전을 보장할 수 없다.

합의 조건을 고정한 뒤 새 검증 문장18개를 준비하고 같은 네 방식을 비교했다.
추론 전에 cases.json과 SHA-256을 저장했다. 학습43개나 임계값을 이 검증 결과로
수정하지 않았다. `results/goal_router_validation`의 모든 결과를 보존한다.

| 방식 | 전체 맞음 | 지원 목표 맞음 | 잘못된 초안 |
|---|---:|---:|---:|
| 최근접13 | 8/18 | 5/12 | 6 |
| 최근접43 | 11/18 | 6/12 | 4 |
| 선형 분류43 | 10/18 | 5/12 | 2 |
| 합의43 | 11/18 | 5/12 | 0 |

합의 방식은 지원12건 중5건만 처리하고7건은 보류했다. 미지원/모순/모호한6건은
모두 보류했다. 거절을 지원 작업의 성공으로 계산하지 않았다. 제안한5건은 모두
기대 목표와 일치했지만 표본이 작아 일반적인 오답률0%의 증거는 아니다.
역방향 목표의 표현 성공도 해당 리눅스 작업 절차가 학습됐다는 증거는 아니다.
조건부 복사, 추가 쓰기, 프로세스/네트워크는 이번 목표 분류 범위 밖이다.

`results/goal_router_consensus_saved`는 같은18건의 저장 후 재검사이며 독립 표본에
추가하지 않는다. 처음 재검사 스크립트가 name 없는 사례에서 KeyError를 냈고,
사례 번호로 식별하도록 수정했다. `goal_router_consensus_replay`의 부분 기록도 보존했다.

## 실제 실행 경로

`results/goal_router_execution_replay`: 기존 진단 문장
‘Duplicate source.bin into target.bin, preserving source.bin.’의 초안을 스크립트
호출자가 기대 조건과 대조·승인했다. 외부 cp 관찰 근거와 텐서 계약을 연결한 뒤
새 파일 바이트2개를 리눅스에서 정확히 복사하고 원본·다른 파일·빈 폴더를 보존했다.
저장 후 두 번째 실행은 관찰/학습 없이 단일 계약 호출이었다. 새 자연어 평가나
실제 인간 승인, 실시간 Gemma 동작의 측정은 아니다. 테스트의 도구 호출 모델은
고정 프로토콜이며 실행 작업 자체는 저장된 텐서 계약이 수행한다.

역할명만 있는 한국어 요청으로 실제 경로를 임의로 연결한 첫 실행 시험은 실행기가
거부했다. 그 에러를 처리 못한 시험용 프로토콜도 실패했다. 경로를 덧붙인 요청은
합의 분류기가 보류했다. 이 실패들은 `goal_router_execution*`의 부분 기록과
`goal_router_failures.json`에 남겼다. 처리 범위가 파일명 표기에 따라 달라지는
문제도 남아 있으며, 통과하는 재시험 문장만으로 이 문제를 해결했다고 하지 않는다.

## 재현

결과 폴더는 기존 기록을 덮어쓰지 않는 새 이름을 사용한다. 기존 컨테이너 하나와
설치된 MiniLM/Gemma를 그대로 사용했으며 이미지/모델 다운로드는 없다.

```powershell
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_router_benchmark --output results/goal_router_stress_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_router_benchmark --validation --output results/goal_router_validation_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_draft --goal-memory --router ridge_consensus --teacher-file experiments/goal_router_training.json --cases-file results/goal_router_validation/cases.json --output results/goal_router_consensus_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m experiments.goal_memory_execution --copy --program results/goal_router_consensus_saved/program.pt --output results/goal_router_execution_recheck
nerdctl exec -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 vectorpro-test python -m pytest -q
```

전체 회귀385 passed (289.81초). 분류기 저장/재로드, 재학습 없는 조회, 불일치 보류,
손상된 가중치/그룹/임계값 거부와 기존 호환성을 포함한다. 전체 회귀 후 실행 시험의
프로토콜과 경로 표기만 수정했으며 리눅스 복사2/2를 추가로 확인했다.

현재 위치는 ‘목표 해석의 오답 억제’ 단계다. 다음 과제는 보류7건의 처리 범위를
넓히되 새 독립 평가의 오답을 늘리지 않는 것, 파일명과 역할의 안정적인 연결,
프로세스/네트워크 목표 확장, 넓은 입력 영역에서 학습된 절차의 정확성을 검증하는
것이다. 범용 자연어 정확성과 임의 프로그램의 동등성은 아직 보장하지 않는다.
