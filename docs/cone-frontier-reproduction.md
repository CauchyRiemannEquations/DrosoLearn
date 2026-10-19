# 같은 수정량의 관측 안전 방향 분석 재현

연구 `cone_frontier_v1`, 완료 태그 `cone-frontier-v1`이다. 부모 결과를 이미 아는 후향적 수학 분석이며 자율 학습 실험이 아니다.

| 항목 | 값 |
|---|---|
| 기준선 | `normalization-loss-v1` / `f1ada01a904283a1036b4dc77b10fb3882194750` |
| 과학 source commit | `70e13961dd17d6f5b70d75bace03c398ec1bc363` |
| 등록 게시 commit | `33184c1691152d2ad982f248126a20f62d4460f0` |
| 등록 SHA-256 | `a7b74b79578623d6f982849de89554f54f3f76ac7caae43d56b89b414c4263db` |
| 설정 SHA-256 | `d10a1b7d249554a2b47d64ce1991012a72afd2acb391898b09ee967ad1dd01ca` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 분석 UTC | 2026-10-19T23:16:28.251179+00:00 → 2026-10-19T23:16:33.046859+00:00 |
| 독립 검산 UTC | 2026-10-19T23:16:44.285404+00:00 |

소스16·부모11파일의 정확한 상대 경로와 SHA-256을 등록했다. 과거 약화 분석의 원자료·독립 검산·완료49파일
snapshot과 더 이전 선택 억제의 전체 replay/source/data 증거를 변경 없이 재사용한다. 새 분석은 선택 억제175배열 중
feedback의6경로를 읽으며 이전24조건 전체의 실패나 원자료를 수정하지 않는다. 새 외부 다운로드·그래프·특징 생성은 없다.

## 저장 결과 독립 검산

비공개 접근 권한과 전체 Git 이력·태그를 가진 새 clone, Python3.12 환경에서 실행한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-cone-check
cd DrosoLearn-cone-check
git checkout cone-frontier-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_cone_frontier.py --config configs/cone_frontier_v1.json --registration results/registration_cone_frontier_v1.json --result-dir results/cone_frontier_v1 --publication-proof results/cone_frontier_publication_proof.json --out scratch/cone-verification.json
```

출력은 존재하지 않는 파일을 지정한다. 새 producer와 solver를 import하지 않는 검산기가 저장된 D와 실제 획득
부호로 제약·목적을 재구성하고 원래 특징 공간에서 활성 제약을 별도로 열거한다. 비음수 승수의 독립 쌍대 상한으로
최적값을 확인한다. 퇴화점의 벡터 bytes가 같다는 조건 대신 방향의 안전·달성값·증거를 검사한다.
제약 원래 행의 cutoff1e-12, 고정 수치 허용 오차1e-8을 유지한다. 실제 관측과 full-information 참고를 구분한다.
15배열·1,260행·42조건/후보 가족의 전체 요약과672,000효과를 검산한다. 과거 전체 학습 replay는 원래 동결 증거를 재사용하며 새 학습으로 세지 않는다.

## 분석을 처음부터 재현

별도 새 clone에서 등록 게시 commit `33184c1691152d2ad982f248126a20f62d4460f0`으로 checkout한다. 이 판에는 새 결과 폴더가 없다.
위 환경을 설치하고 다음을 임시 `copy-proof.py`로 저장해 원 게시 증거를 bytes 그대로 복사한다.

```python
from pathlib import Path
import subprocess
name = 'results/cone_frontier_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'cone-frontier-v1:' + name]))
```

```text
.venv\Scripts\python.exe copy-proof.py
.venv\Scripts\python.exe scripts/run_cone_frontier.py run --config configs/cone_frontier_v1.json --registration results/registration_cone_frontier_v1.json --publication-proof results/cone_frontier_publication_proof.json --out results/cone_frontier_v1
.venv\Scripts\python.exe scripts/verify_cone_frontier.py --config configs/cone_frontier_v1.json --registration results/registration_cone_frontier_v1.json --result-dir results/cone_frontier_v1 --publication-proof results/cone_frontier_publication_proof.json --out results/cone_frontier_v1/verification.json
```

부모 학습을 다시 실행하지 않는다. 같은 저장된 g·관측 상태를 쓰며 새 방향이 이후 선택에 영향을 주지 않는다.
NoGo 영향/이동은0, 최적화 case index는−1이다. 현재 실제 Go 보상 뒤의 부호만 관측 후보에 전달한다.
새 학습·난수·행동·내부/출력 가중치 갱신·모델 생성은0회다. 재현의 UTC·metadata·NPZ 컨테이너 bytes는 달라질 수 있지만
같은 환경의 수치·캐시 키·행·판정은 일치해야 한다. 새 표본이나 성공을 찾는 반복으로 세지 않는다.

## 검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_cone_geometry.py tests/test_cone_frontier.py tests/test_cone_envelope.py tests/test_cone_frontier_verify.py -q
```

새165개와 기존69개(`test_selective_spillover.py`30, `test_selective_geometry.py`39), 총234개 검사가 통과했다.
최종 JUnit과16소스 해시는 `results/cone_frontier_prefreeze_validation.json`에 있다. 합성 예제와 작은 가짜 경로만 사용했다.
개발 중 예제의 마지막 보상 발견 시행 예상값을 바로잡았으며, 최종 검사만 중복 없이 센다.
`cone-frontier-protocol.md`와 등록이 실제 집계 전에 고정돼 있다.

그림은 동결 `drosolearn.cone_frontier.render`로 독립 검산 뒤 생성했다. `cone_frontier_figures_v1`의
표시 수치·provenance가 요약·배열·검산·renderer·PNG를 해시로 연결한다. 60개 표시 비율을 원 요약과 대조했다.
비율 분모는 이미 정규화된 저장 방향의 같은-g 점수다. 직전 약화 분석 raw 목표의100과 섞지 않는다.

`results/cone_frontier_baseline_preservation.json`은 기존18,812파일·2,105,281,712bytes의 **Git blob 해시 전수 대조**다.
새 literal-byte전수 대조라고 부르지 않는다. 이전 literal증거는 원본 그대로 남겼다. 완료 archive는 이후 기존 경로 Git 차이와
새 source/parent/result/doc/figure 해시를 결합한다. 기존 파일 변경은 README·로드맵·상태 JSON 세 색인뿐이다.

관측 안전 제약·전체 정보 안전 참고·미관측 실제 방해, pooled비율·시드평균, 합산 개선·시행별 동시 개선,
선택 점수·정답률·실제 학습 성적을 [쉬운 결과](cone-frontier-findings.md)에서 구분한다.
