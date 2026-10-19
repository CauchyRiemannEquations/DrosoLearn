# 큰 결과 파일의 바이트 보존 저장

간섭·순서 연구의 원 `results/interference_order_v1/arrays.npz`는 104,904,900바이트다. [GitHub의 일반 Git 단일 파일 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)은 100 MiB(104,857,600바이트)로 원 결과가 47,300바이트 크다. 원 NPZ를 재압축·축소·교체하지 않는다.

원본은 로컬에 그대로 두고 두 연속 바이트 조각(67,108,864와 37,796,036바이트)을 일반 Git으로 저장한다. [분할 manifest](../results/interference_order_v1/arrays.npz.parts.json)에 원 경로·원 크기·원 SHA256 및 순서가 있는 각 조각의 경로·크기·SHA256을 기록했다. 두 조각을 그대로 연결하면 원본과 정확히 같은 바이트가 된다. 숫자 배열을 다시 직렬화하는 방식이 아니다. 원 등록·원 artifact hash·독립 검산·실패는 변경하지 않는다. LFS나 새 외부 유료 저장소를 사용하지 않는다.

새 결과 폴더의 `.gitignore`는 이 원 NPZ 한 경로만 제외한다. 기존 `.gitignore`·`.gitattributes`·완료 소스를 바꾸지 않는다. 결과의 metadata와 artifact hash는 여전히 원 NPZ를 가리키며 분할 파일은 추가 저장 포장이다.

원 등록·실험의 검산기는 기존 그대로다. 새 Phase 5 소스에서만 Git 바이트 검사의 저장 형태를 보완한다. **오직 위 한 부모 경로**의 경우, Git의 원 artifact hash·manifest·각 조각과 현재 파일을 대조하고, 연결된 길이·SHA가 기존 부모 해시 및 원 NPZ와 같아야 허용한다. 다른 모든 소스·부모 검사는 동결된 공통 함수를 그대로 사용한다. 예외를 다른 경로나 다른 해시로 확대하지 않는다. 독립 검산기는 생산자를 import하지 않고 같은 저장 검증을 별도로 구현한다.

새 clone에서는 다음 명령으로 누락된 원 파일을 복원한다. 기존 파일이 있으면 정확한지 확인만 하며 덮어쓰지 않는다. 다른 바이트라면 중단한다. 원 파일 없이 일반 공통 실행기를 바로 사용하면 부모 검사가 거부하는 것이 정상이다.

```powershell
.venv/Scripts/python.exe -m drosolearn.phase5_report materialize-parent
```

복원은 `xb` 배타적 생성이며 원 과학 실행·새 시드가 아니다. final 태그의 검산·재현 명령은 [재현 안내](phase5-reproduction.md)를 따른다. Phase 5 등록·실행은 이 저장 보완이 포함된 별도 모듈 진입점을 사용하며 공통 기존 코드와 결과를 수정하지 않는다. 최종 보존 감사는 태그 안의 두 조각을 연결한 바이트와 원 artifact hash도 대조한다.
