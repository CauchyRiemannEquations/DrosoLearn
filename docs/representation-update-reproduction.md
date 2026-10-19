# 입력 표현·학습 갱신 진단의 재현

이 연구는 완료한 Phase 5 종합 뒤에 별도 등록한 후속 연구다. 이전 등록·실패·코드를 바꾸지 않는다.
태그 `representation-update-v1`에서 원자료·검산·보고서를 읽을 수 있다.

## 고정된 실행판과 데이터

| 항목 | 값 |
|---|---|
| 완료 기준선 | `phase5-synthesis-v1` / `5978fd127c0f0787a5f2068c9d0e5fc74b35e62c` |
| 새 source commit | `6f112178879425f8c91c30daee261e81198e1c73` |
| 등록 게시 commit | `21c7e366d4efcb75fbd432c9a50bbe39f3f978b4` |
| 등록 SHA-256 | `22fcec19d636cfde84e2ccde56d599226b269e8ae485d3b20ef12dae74aabb3a` |
| 설정 SHA-256 | `4cd21bf8d8f6f318ee4c40bab7a3835dd7a8e1332521d00a7ca32034fff24637` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 계산 UTC | 2026-10-19 04:23:12.780427 → 04:23:18.572119 |
| 독립 검산 UTC | 2026-10-19 04:23:40.669955 |

그래프는 저장소 안의 `data/flywire_783_mb_left_kc512_s701` 네 원본 파일이다.
KC 512·MBON 48·KC→MBON 1,474연결이며 활성 노드는 489 KC·29 MBON이다.
입력은 새 유한 catalog와 이전 pattern catalog다. 입력 생성은 순수 명세이고 성능 적합을 사용하지 않는다.
사람 원자료·새 내려받기·대형 맥락 실험 NPZ는 필요하지 않다.

`results/registration_representation_update_v1.json`에 소스 20·부모 11·그래프 4·입력 2개의
정확한 파일 목록과 SHA-256이 있다. `requirements-lock.txt`와 Python 3.12 환경을 사용한다.
같은 source inventory가 유지돼야 하며 새 모듈을 추가했다는 이유로 과거 source 목록을 고치지 않는다.

## 이미 저장한 결과의 검산

완료 태그의 새 clone에서 다음 환경을 만든다. 이 예시의 `python`은 Python 3.12여야 한다.

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
```

기존 `verification.json`을 덮어쓰지 않도록 새 검산 경로를 지정한다.
`scratch`가 없다면 Python 또는 셸로 새 폴더를 만든다.

```text
.venv\Scripts\python.exe scripts/verify_representation_update.py --config configs/representation_update_v1.json --registration results/registration_representation_update_v1.json --result-dir results/representation_update_v1 --publication-proof results/representation_update_publication_proof.json --out scratch/representation-verification.json
```

이 검산은 저장된 그래프와 난수 주소에서 모든 수치를 재구성하며, 새 producer와 기하 helper를 import하지 않는다.
기존 결과·원 등록·원 검산을 수정하지 않는다. 등록 원문/코드 해시와 시간 순서도 검사한다.

자동검사는 다음 네 파일이다. envelope 검사는 작은 가짜 Git 저장소만 사용하므로 실제 실험을 시작하지 않는다.

```text
.venv\Scripts\python.exe -m pytest tests/test_representation_geometry.py tests/test_representation_update.py tests/test_representation_envelope.py tests/test_representation_update_verify.py -q
```

## 학습 자체를 동일 설정으로 다시 실행하려면

**원 작업 폴더에서 실행하지 않는다.** runner는 고정 결과 경로가 이미 있으면 거부한다.
새 clone 또는 별도 checkout을 만들고 **등록 게시 commit `21c7e366d4efcb75fbd432c9a50bbe39f3f978b4`**으로 이동한다.
그 판에는 코드·입력·등록이 있고 과학 결과 폴더는 아직 없다. 전체 Git 이력과 기준선 태그가 필요하다.

완료 태그에서 `results/representation_update_publication_proof.json`의 정확한 bytes를
새 checkout의 같은 경로로 복사한다. 예를 들어 아래 내용을 `copy-proof.py`로 저장해 새 checkout에서
실행하면 셸의 문자열/줄바꿈 재인코딩을 피할 수 있다. 이는 기존 원격 게시 증거의 복사이며 새 등록이 아니다.

```python
from pathlib import Path
import subprocess
path = 'results/representation_update_publication_proof.json'
with Path(path).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'representation-update-v1:' + path]))
```

위 환경 설치 뒤 같은 고정 예산으로 실행한다.

```text
.venv\Scripts\python.exe scripts/run_representation_update.py run --config configs/representation_update_v1.json --registration results/registration_representation_update_v1.json --publication-proof results/representation_update_publication_proof.json --out results/representation_update_v1
.venv\Scripts\python.exe scripts/verify_representation_update.py --config configs/representation_update_v1.json --registration results/registration_representation_update_v1.json --result-dir results/representation_update_v1 --publication-proof results/representation_update_publication_proof.json --out results/representation_update_v1/verification.json
```

재실행 UTC와 metadata/ZIP 컨테이너 bytes는 달라질 수 있다. 같은 환경에서 배열 수치·행동·보상·
기준 판정이 일치하는지를 독립 검산으로 확인한다. 새 실험으로 세거나 성공을 고르기 위한 반복으로 쓰지 않는다.
현재 작업에서는 과학 학습을 한 번 실행하고 독립 재계산 한 번을 수행했다.

## 보존된 파일

- `results/representation_update_v1`: 등록 사본·metadata·559개 수치 배열·48,000행 CSV·요약·5원본 hash·독립 검산.
- `results/representation_update_prefreeze_validation.json`: 신규 고유 162검사와 관련 기존 103검사, 실행 전 source hash.
- `results/representation_update_prefreeze_tests`: JUnit 원문. 개발 중 중복 검사는 합산하지 않는다.
- `results/representation_update_prefreeze_whitespace_correction.json`: 등록 전 두 파일의 EOF 공백만 정리한 내역.
- `results/representation_update_figures_v1`: 두 정적 그림·그린 수치·출처 hash.
- `results/representation_update_archive_validation.json`: Phase 5 기준선의 모든 기존 tracked 파일을
  세 가변 색인 외에 원문 그대로 보존했는지 확인한 기록. 과거 학습을 다시 실행하지 않는다.

보관 확인 도구는 `scripts/check_representation_archive.py`다. 기본 결과 경로에 기존 보관 기록이 있으면
덮어쓰기를 거부한다. 원 보관 기록의 파일별 hash로 내용을 검사할 수 있다.
전체 비용 통제·새 문자열 일반화·인간 유사성은 이 재현의 대상이 아니다.
