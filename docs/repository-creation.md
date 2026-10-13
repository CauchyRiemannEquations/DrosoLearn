# 독립 저장소 생성 절차

로컬 `DrosoLearn` 프로젝트는 기존 DrosoMem의 작업트리와 Git 이력을 공유하지 않는 신규 저장소다. 원본은 읽기 전용 snapshot으로 감사했고 source commit을 보존했다. GitHub에 생성하는 경우 기본은 private로 하며 visibility 변경은 이후 소유자의 선택이다.

인증·생성 권한이 있는 환경의 재현 절차:

```powershell
Set-Location <DrosoLearn 프로젝트 절대 경로>
git init -b main
git add .
git commit -m "Initialize independent DrosoLearn research pipeline"
gh auth status
gh repo create DrosoLearn --private --source . --remote origin --push
```

이미 로컬 Git이 초기화되어 있거나 remote가 존재하면 해당 단계를 반복하지 않는다. 동일한 이름의 GitHub 저장소가 존재하면 자동 덮어쓰기·삭제하지 않고 새 이름을 지정한다. 인증이 없다면 `gh auth login` 후 위 생성 명령을 실행할 수 있으며 로컬 프로젝트·실험·테스트는 GitHub 없이 작동한다. 외부 서비스에 올리는 파일은 출력 프로젝트의 코드·문서·소규모 데이터·결과만이며 scratch 환경과 원본 snapshot은 포함하지 않는다.

실제 생성 여부와 URL은 `repository-status.json`에 기록한다. 저장소 생성은 생물학적 실험이나 결과의 확증·학술출판을 의미하지 않는다.
