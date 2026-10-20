# 초반 보호 후 균형 학습의 재현

연구 `gated_learning_v1`, 완료 태그 `gated-learning-v1`이다. 앞선 저장 경로 분석을 알고 전환 규칙을 설계했다. 새 코호트의 학습·성적 집계는 성공한 등록 게시 뒤 처음 실행했다. 기존 실패의 시드·예산·문턱을 바꾸지 않았다.

| 항목 | 값 |
|---|---|
| 완료 기준선 | `balanced-pathways-v1` / `f291f1f22e7ce83a0c836924530ca93e9bc12d33` |
| 과학 source commit | `6c3f709263c8fc7bea8ad52efbafca96afc5d188` |
| 등록 게시 commit | `100434244591e454a495dc77b844058fa2031185` |
| 등록 SHA-256 | `fa4791c1c32e3c13b9fbf16184b9e8a5f092647075a8c7b2f32ff389530acce5` |
| 설정 SHA-256 | `500e520514f3b9bf71c3aefd029748d49eacfbeba63517158d3c2d68aa4d2807` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 게시 확인 UTC | 2026-10-20T00:25:29.956283+00:00 |
| 실제 실행 UTC | 2026-10-20T00:25:49.255074+00:00 → 2026-10-20T00:27:30.379415+00:00 |
| 독립 검산 UTC | 2026-10-20T00:29:27.432014+00:00 |

소스39·선행11·그래프4·입력2파일의 경로·SHA-256을 등록했다. FlyWire v783 KC512/MBON48/1474연결의 고정 부분 투사와 원 출처·입력 catalog·인코더 규칙을 유지한다. 배선3×표현2×방법5의30조건, 새 시드17601..17610, namespace202610202, 조건당400시행, 체크포인트0/50/100/200/400, 학습률0.1이다. actual movement 상한은 시행0.1·경로40·제곱합4이며 실제 소비량·최적화 비용은 동일하지 않다.

## 저장 결과만 독립 검산

비공개 저장소 접근 권한과 전체 Git 이력·태그가 있는 새 clone에서 Python3.12로 실행한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-gated-check
cd DrosoLearn-gated-check
git checkout gated-learning-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_gated_learning.py --config configs/gated_learning_v1.json --registration results/registration_gated_learning_v1.json --result-dir results/gated_learning_v1 --publication-proof results/gated_learning_publication_proof.json --out scratch/gated-verification.json
```

출력 파일은 없어야 한다. 독립 검산기는 새 과학 생성자·게이트 학습자를 import하지 않는다. 그래프·입력·특징·난수 순서·120,000행동과 보상·기억·전환·출력 수정·6,000평가·195배열·CSV84열과 세 판정을 재구성한다. 실제 관측 상태720개는 원 특징 공간의 별도 least-squares/비음수 쌍대 상한으로 다시 검증한다. 퇴화점에서 최적 벡터 bytes의 동일성을 강제하지 않는다. 검산용 RNG와 수치 재구성은 새로운 연구 표본이나 추가 학습으로 세지 않는다.

## 실제 학습을 처음부터 재현

다른 새 clone에서 등록 게시 commit `100434244591e454a495dc77b844058fa2031185`으로 checkout한다. 이 판에는 실제 결과 폴더가 없다. 위와 같은 환경을 설치한다. 다음 내용을 임시 `copy-proof.py`로 저장해 완료 태그의 게시 증거 bytes만 복사한다.

```python
from pathlib import Path
import subprocess
name = 'results/gated_learning_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'gated-learning-v1:' + name]))
```

```text
.venv\Scripts\python.exe copy-proof.py
.venv\Scripts\python.exe scripts/run_gated_learning.py run --config configs/gated_learning_v1.json --registration results/registration_gated_learning_v1.json --publication-proof results/gated_learning_publication_proof.json --out results/gated_learning_v1
.venv\Scripts\python.exe scripts/verify_gated_learning.py --config configs/gated_learning_v1.json --registration results/registration_gated_learning_v1.json --result-dir results/gated_learning_v1 --publication-proof results/gated_learning_publication_proof.json --out results/gated_learning_v1/verification.json
```

완료 결과가 있는 checkout에서 실행해 덮어쓰지 않는다. 고정 결과 경로의 배타 생성은 보호 장치다. 재현 metadata UTC·NPZ 컨테이너 bytes는 달라질 수 있으나 같은 환경의 실제 수치·선택·보상·게이트·캐시 키·판정은 일치해야 한다. 옛 연구를 재실행할 필요는 없다.

## 검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_gated_learning.py tests/test_gated_envelope.py tests/test_gated_learning_verify.py tests/test_balanced_learning.py -q
```

새181개(학습47·무결성48·독립 검산86)와 기존 관련61개를 합친242개가 통과했다. 정확한 명령·테스트 집합은 실제 JUnit 및 `results/gated_learning_prefreeze_validation.json`과 연결해 확인한다. 개발 검사에는 합성 특징·허구 시드만 썼고, 새 실제 코호트의 성적을 보고 조정하지 않았다.

최초 소스 커밋 `32de2f9d5165108d711f2c7d9ae3d431059d19ca`이 동결 보조 작업 완료 전에 시작돼 최종 검산 테스트 bytes가 Git에 반영되지 않은 경합이 있었다. 등록이 이 불일치를 거부했다. 첫 커밋과 `results/gated_learning_prefreeze_finalization.json`을 보존했으며, 최종39소스·242검사를 확인하고 커밋한 뒤 처음 성공한 등록을 게시했다. 이전에 실제 새 코호트 그래프·학습·집계는0회였고 설정·시드·예산·문턱·모델 소스는 바꾸지 않았다.

독립 검산 후 그린 네 그림·`plotted_values.json`·`provenance.json`은 원 배열·요약·검산·renderer 해시를 연결한다. 기존 그림을 덮어쓰지 않고 완료 태그에서 새 폴더로 재생성할 수 있다. 결과만 읽고 추가 학습을 하지 않는다.

```text
.venv\Scripts\python.exe scripts/render_gated_learning.py --out scratch/gated-figures
```

`scratch`는 먼저 만들고 그림 출력 폴더는 없어야 한다. 그림에는30조건 학습곡선, 주 조건5방법의 성적·방해·실제경로, 누적 미관측 방해와10시드 전환 시점이 있다. 영어 그림 용어 Shared/Isolated/Selective/Balanced/Gated는 [쉬운 결과](gated-learning-findings.md)의 다섯 방식과 같은 순서다.

`results/gated_learning_baseline_preservation.json`은 기준선18,918파일·2,182,824,549bytes의 현재 raw bytes를 Git blob 해시로 전수 대조한 기록이다. 새 literal-byte 전수 비교라고 부르지 않는다. 완료 archive는 이후 기존 경로 diff와 동결 source/parent/data/result/doc/figure 해시를 함께 검사한다. 기존 파일 변경은 README·로드맵·상태 JSON 세 색인뿐이다.

자동검사·실제 실행·독립 산술 검산의 성공은 과학적 가설의 성공과 다르다. 이번 H_protection은 통과, H_gain·H_acquisition은 미지지다. 원자료·실패·시드·예산을 그대로 보존한다. 내부 가중치 고정, 출력 계수와 보상 기억의 변화, 미관측 연습 입력과 새 문제 일반화, 계산 시드와 생물학적 개체의 차이를 유지한다.
