# Claude 작업 시작 지침

이 프로젝트의 공통 협업 지침은 [AGENTS.md](AGENTS.md)에 있다.
작업 전에 반드시 읽고 동일하게 적용한다. 중복된 규칙을 이 파일에 따로 관리하지 않는다.

추가로 읽을 문서:

- [초기모델 명세](docs/INITIAL_MODEL.md)
- [첫 완성 목표: 리눅스 사용](docs/LINUX_TARGET.md)
- [목록·문자열·JSON과 학습된 순회](docs/STRUCTURED_DATA.md)
- [여러 결과 전달의 초기 기반](docs/RECORD_RESULTS.md)
- [프로세스 실행과 학습된 출력 전달](docs/PROCESS_RESULTS.md)
- [순차 프로세스 연결과 실패 분기](docs/PROCESS_CHAIN.md)
- [작업 인수인계](docs/HANDOFF.md)
- [구현·실험 안내](README.md)
- [LLM 어댑터와 근거의 한계](docs/LLM_ADAPTER.md)
- [다른 OS의 동일 파일 검증 절차](docs/PORTABILITY.md)
- [변경별 테스트와 재현 방법](docs/TESTING.md)
- [에이전트 반복 실행 비용 실측](docs/AGENT_COST.md)
- [텐서 카탈로그·검색 정확도 검증](docs/TENSOR_CATALOG.md)
- [정식 텐서 저장·LLM 카탈로그 연결](docs/CATALOG_AGENT.md)
- [공통 기능 계약과 직접 호출](docs/FUNCTION_CONTRACTS.md)
- [계약 초안 학습·검증·등록](docs/CONTRACT_LEARNING.md)

특히 **기존 `vectorpro-test` 컨테이너 하나를 `nerdctl exec`로 재사용**한다.
새 컨테이너나 이미지를 만들지 않는다. 작업별 절차를 사람이 작성해 놓고 학습
결과라고 설명하지 않는다. 현재 구현 범위와 아직 필요한 확장을 명확히 구분한다.

## 2026-10-05: 리눅스 기능 통합

스트리밍 파이프, 프로세스 전후 전체 상태 관찰, HTTP 및 시스템 조회와 조합 학습을 추가했다. 다섯 기능의 학습/검증과 저장 후 실제 리눅스 13건 검증을 통과했다. 실행 기능과 습득 절차의 구분, 재현 명령, 현재 범위와 남은 과제는 docs/LINUX_BUNDLE.md를 참조한다. 최종 증거는 results/linux_bundle_final_verified에 있다. 전체 회귀 결과는 docs/HANDOFF.md에 기록한다.
