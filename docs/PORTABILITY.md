# 동일 프로그램 파일의 이식성 검증

학습한 파일은 `results/initial_model/program.json`이다. 숫자 테이블과 호출·분기·
반복 텐서만 담고 있으며 실제 OS 핸들·실행 루트는 새 런타임에서 제공한다.
검사기는 학습 없이 파일을 불러와 숫자 계산·파일 변환·조건부 반복·여러 반복·
한글 경로를 확인한다. SHA-256이 같아야 동일한 프로그램 파일을 검증한 것이다.

현재 프로젝트 규칙은 모든 테스트를 기존 `vectorpro-test` 컨테이너에서만
실행하는 것이다. 이 규칙 아래에서 확인된 플랫폼은 Linux다. Windows/macOS에서
실제로 실행했다고 주장하지 않는다. 해당 호스트와 호스트 테스트 허용이 확보된
경우 아래 절차를 적용한다. 새 컨테이너나 이미지는 필요하지 않다.

1. 프로젝트 소스와 같은 `program.json` 파일을 대상 컴퓨터로 가져간다.
2. 프로젝트가 선언한 Python/PyTorch 런타임을 설치한다.
3. 재학습 없이 검사기를 실행하고 결과 파일을 보관한다.

```text
python scripts/check_portability.py --program results/initial_model/program.json --output portability-HOST.json
```

Linux 검증 결과는 `results/initial_model/portability-linux.json`에 있다. 대상 OS의
결과와 프로그램 SHA-256을 비교한다. 플랫폼별 파일 시스템 특성이 다르므로
이 검사 성공은 모든 경로·모든 OS 동작의 동일성을 증명하는 것은 아니다.
호스트 경로 입력은 이식 가능한 UTF-8 상대 경로로 통일하며, 역슬래시·드라이브
접두사·상위 디렉터리 접근을 거절한다.
