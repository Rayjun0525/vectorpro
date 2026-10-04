# Claude 작업 시작 지침

이 프로젝트의 공통 협업 지침은 [AGENTS.md](AGENTS.md)에 있다.
작업 전에 반드시 읽고 동일하게 적용한다. 중복된 규칙을 이 파일에 따로 관리하지 않는다.

추가로 읽을 문서:

- [초기모델 명세](docs/INITIAL_MODEL.md)
- [첫 완성 목표: 리눅스 사용](docs/LINUX_TARGET.md)
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
