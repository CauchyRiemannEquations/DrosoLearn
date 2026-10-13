# DrosoLearn

**Exploring the Foundations of Mathematical Learning Through a Drosophila Connectome**

초파리 커넥톰 제약을 가진 계산 모델에서 수량·기초 규칙을 배우는 과정, 일반화, 보상 수정과 학습 간섭을 조사한다. 인간의 기초 수량 인지·수학 학습에 대해 검증 가능한 가설을 만드는 것이 장기 목표다. 커넥톰의 우수성이나 인간과 동일한 신경 메커니즘을 전제하지 않는다.

[DrosoMem](https://github.com/CauchyRiemannEquations/DrosoMem)은 완료된 독립 연구다. 원본 저장소·실험을 변경하지 않고, 특정 커밋의 소규모 데이터를 출처와 함께 재사용한다. 기존 M11 실패 및 이후 제한적 내부 가소성 연구의 부정적 기능 결과를 연구 출발점에 포함했다.

## 현재 구현

- 실제 FlyWire FAFB v783 부분집합의 KC→MBON 1,474쌍을 이용한 one-hop projection. 기존 recurrent reservoir의 복제는 아니다.
- **추상적 수량 보상 연합**: `n/8` 입력이 수량·순서를 이미 제공한다. 시각 수량 추출·LC11 회로·수학 기호 이해는 검증하지 않는다.
- 2대4 습득→동일 모델 보상 반전. 인공 nuisance OOD와 미학습 수량쌍 probe는 분리 평가한다.
- 원래/재배선 × 고정/가소성, 일반 random sparse ANN, frozen·random reward·internal-only: 9조건.
- 전향적 로컬 동결, 다중 seed, 원자료·가중치·활동 checkpoint, 자동 그림 및 별도 검증.

초기 결과와 테스트 상태는 [초기 실행 보고서](results/smoke_v1/report.md), [독립 검증](results/smoke_v1/verification.json)에서 확인한다. 3시드 smoke는 확증·생물학적 검증이 아니다.

## 설치와 실행

Python **3.12**에서 저장소 root를 작업 디렉터리로 사용한다. Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q

# 기존 등록의 source/config 해시가 동일하면 기존 동결로 새 디렉터리에 재현
.\.venv\Scripts\python.exe -m drosolearn.cli run --config configs/smoke.json --registration results/registration_smoke_v1.json --out results/my_smoke_replay
.\.venv\Scripts\python.exe -m drosolearn.cli verify --out results/my_smoke_replay
```

Linux/macOS는 `python3.12 -m venv .venv`와 `.venv/bin/python`으로 실행한다. BLAS/platform에 따른 부동소수점 차이가 있을 수 있으며 버전·platform과 tolerance를 보존한다. 각 run은 약 21,600개의 학습 시행과 27 seed-condition을 포함한다. 결과 경로가 존재하면 덮어쓰기를 거부한다.

코드·데이터·프로토콜·설정을 변경할 때는 이유를 amendment로 남기고 **새 등록**을 먼저 만든다:

```powershell
.\.venv\Scripts\python.exe -m drosolearn.cli register --config configs/smoke.json --out results/registration_NEW.json
.\.venv\Scripts\python.exe -m drosolearn.cli run --config configs/smoke.json --registration results/registration_NEW.json --out results/NEW
.\.venv\Scripts\python.exe -m drosolearn.cli verify --out results/NEW
```

`configs/confirmatory.json`은 별도 10시드 계획이며 초기 결과를 본 후 수정한 설계의 무조건적 확증으로 사용하지 않는다. 최초 실행 전에 새로운 source/protocol 동결과 수정 내역을 확인한다.

## 연구 문서

| 문서 | 내용 |
|---|---|
| [핵심 질문](docs/research-question.md) | 다섯 연구 질문, 반증과 Level 1–5 |
| [DrosoMem 감사](docs/drosomem-audit.md) | 현재 구현·실험 결과·실제 데이터 범위·재사용 판단 |
| [문헌 검토](docs/literature-review.md) | 지정 논문 3편과 추가 일차 연구의 실제 근거 |
| [기술 설계](docs/technical-design.md) | 입력·회로·가소성·로그 인터페이스와 선택 이유 |
| [사전 프로토콜](docs/experimental-protocol.md) | 입력 분할·보상·대조군·고정 판정 기준 |
| [로드맵](docs/research-roadmap.md) | Phase 0–5와 다음 단계의 gate |
| [인간 비교](docs/human-learning-comparison.md) | 공통 측정 지표·공개 데이터·차이와 한계 |
| [LIMITATIONS](LIMITATIONS.md) | 추상 입력·모델·생물학·교육적 해석 경계 |

## 데이터·재현·GitHub

`data/`에 고정된 작은 원본 subset을 동봉한다. 큰 전뇌 자료나 원시 synapse point cloud는 다운로드하지 않았다. [파일 출처 manifest](data/source-manifest.json), [데이터 조건](data/LICENSE.md), 원본 `provenance.json`과 해시를 함께 보존한다. 새 Python 코드는 작성한 구현이며 DrosoMem runtime 코드를 복사하지 않았다.

`results/`에는 실패를 포함한 결과와 메타데이터만 기록한다. 학습 평가가 좋지 않더라도 예산을 늘려 같은 가설의 성공을 찾지 않는다. 다음 실험은 실패 원인과 입력·가소성·자원의 가설을 분리해 등록한다.

새 GitHub 저장소 생성·배포 절차는 [repository-creation.md](docs/repository-creation.md)에 있다. DrosoMem에 push하거나 원본 결과를 교체하지 않는다. 소프트웨어는 [MIT](LICENSE), FlyWire 파생 데이터는 별도의 CC BY-NC 4.0 조건을 따른다.
