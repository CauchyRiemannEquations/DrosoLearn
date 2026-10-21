# U/P 실제 경로 저장 분석 재현

완료 태그 `objective-pathways-v1`의 별도 checkout에서 원자료와 검산 결과를 확인한다. 기록 원본은 `objective-learning-v1`이며 재학습하지 않는다. 현재 CLI는 고정된 등록/출력 경로를 요구하므로 완료 폴더가 있는 checkout에서 새 경로만 넘겨 재실행할 수 없다.

소스 commit: `3783491b82f61f1d956756b356e09cbae1dcbd58`. 등록 게시: `0d2bead26dc838200bca0cbf4dede2f2c0b15765`.
등록 SHA-256: `2621e15e31907a6d414f92cc7830944e3470d9f77128a22902413c69a8f3926a`.

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest tests/test_objective_pathways.py tests/test_objective_pathways_envelope.py tests/test_objective_pathways_verify.py tests/test_objective_learning.py tests/test_self_blend_factors.py -q
```

291개 고유 검사이며 새 176개와 기존 관련 115개다. 최초 실행 전 프로토콜·설정·소스 25개·부모 16개·사전 검사 영수증을 고정하고 비공개 원격 등록의 동일 bytes를 확인했다. 아래 명령은 최초 실행 기록이며, 원 폴더가 존재하면 fail closed 한다.

```powershell
python scripts/run_objective_pathways.py register --config configs/objective_pathways_v1.json --out results/registration_objective_pathways_v1.json
python scripts/run_objective_pathways.py run --config configs/objective_pathways_v1.json --registration results/registration_objective_pathways_v1.json --publication-proof results/objective_pathways_publication_proof.json --out results/objective_pathways_v1
python scripts/verify_objective_pathways.py --config configs/objective_pathways_v1.json --registration results/registration_objective_pathways_v1.json --result-dir results/objective_pathways_v1 --publication-proof results/objective_pathways_publication_proof.json --out results/objective_pathways_v1/verification.json
python scripts/render_objective_pathways.py
python scripts/check_objective_pathways_archive.py
```

위 명령은 최초 실행의 기록이다. 완료 태그의 원 결과를 보존한 상태에서 독립 verifier의 `--out`에 아직 존재하지 않는 별도 영수증 경로를 지정하면 읽기 전용 검산을 할 수 있다. archive 검사기는 완료 영수증이 있으면 덮어쓰지 않고 종료한다. 전체 집계를 새 경로에서 다시 수행하려면 별도 프로토콜/버전에서 고정 경로를 명시적으로 정해야 하며, 완료 폴더나 이전 소스를 지우거나 수정해 덮어쓰지 않는다. 실제 최초 원자료 집계는 한 번, 독립 산술은 별도 코드로 한 번 수행했다. 새 학습·난수·방향 최적화 0회이며 완료 부모의 192,000 실제 선택/12,989 solver를 재실행하지 않는다. 부모 원자료의 기존 전체 독립 검산과 byte gate를 보존한다.

배열은 16개, CSV는 2,160행·15열이다. 공통 후반 양쪽 Go의 state/scalar/direction 8셀을 정답 확률의 정확 sigmoid 식으로 대입해 순/양/음 질량의 Shapley 배분을 확인한다. 한쪽 실제 Go는 별도 나머지로 남겨 최종 성적 차이와 회계가 닫히는지 검사한다. 전체 48조건/480경로의 선택·보상·기억·수정량·미관측 방해와 all/pre/post·source/action/reward 34,560칸도 독립 계산한다.

환경 버전은 metadata.json에 기록했다. 원 저작권·FlyWire 데이터 출처/버전/해시·DrosoMem 활용 한계는 부모 provenance와 원 문서를 그대로 따른다. 성공 기준이나 부모 실패를 바꾸지 않는다. 실험 원본·소스·설정·실패는 변경하지 않으며 세 상태 인덱스만 최신 경로를 가리키게 했다.
