# Marković 과제 규칙과 기록 코드의 출처 확인

작성일: 2026-10-15. 이 문서는 원논문과 공개 프로그램을 읽은 **과제 의미 감사**다. 인간 원본 MAT나 행동 CSV를 새로 읽지 않았고, 선택 빈도·정확도·보상률·반전 경계·학습 곡선을 계산하지 않았다. 앞선 [원본 재현 결과](human-osf-origin-findings.md)는 별도 결과로 보존한다.

## 먼저 쉽게 설명하면

이 실험의 숫자에는 서로 다른 뜻이 섞여 있다. 카드 선택의 `0/1`은 두 선택지의 이름이다. 결과의 `0/1`은 손실과 이득이며, 결과의 `2/3`은 어느 카드가 현재 유리한지 알려 주는 정보다. **결과 3이 결과 1보다 세 배 좋은 보상이라는 뜻은 아니다.** 정보를 요청하면 그 시행에 돈을 얻거나 잃지 않지만 카드 선택 기회를 쓰게 된다.

또한 ‘800번째 시행 뒤 변화’는 반전이 매 800시행마다 일어난다는 뜻이 아니다. 유리한 카드가 여러 차례 바뀌는 가운데, 마지막 200시행에는 **바뀌는 간격의 규칙**이 반대 조건으로 바뀐다. 이 두 종류의 변화를 분리해야 후속 기록을 올바르게 비교할 수 있다.

## 직접 확인한 출처

- Marković, Reiter & Kiebel (2022), *Revealing human sensitivity to a latent temporal structure of changes*, [DOI: 10.3389/fnbeh.2022.962494](https://doi.org/10.3389/fnbeh.2022.962494). [PMC 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9621332/)의 §2·그림 1–2·§4.2.1·§4.3을 확인했다. 논문은 CC BY를 명시한다.
- 논문이 연결한 [공식 OSF 프로젝트 h526v](https://osf.io/h526v/)의 [과제 코드 component 57wr8](https://osf.io/57wr8/), `Experiment` 폴더 `5d08b62443bdc30017e87aa3`. [공식 폴더 API](https://api.osf.io/v2/nodes/57wr8/files/osfstorage/5d08b62443bdc30017e87aa3/)에서 파일 ID·크기·SHA-256·버전을 확인하고 **revision 1**의 작은 MATLAB 텍스트만 받았다. component의 [공식 node API](https://api.osf.io/v2/nodes/57wr8/)가 연결하는 [라이선스 API](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/)는 CC BY 4.0이다.
- 논문 §4.1이 연결한 저자 [pybefit 저장소](https://github.com/dimarkov/pybefit)의 고정 커밋 `813aeb5ad2c7bbed90ecd7d947bce9fe981c9702`. [main formatter](https://github.com/dimarkov/pybefit/blob/813aeb5ad2c7bbed90ecd7d947bce9fe981c9702/examples/temporal_rev_learn/behavioral_data/main/format_behavioral_data.py), [pilot formatter](https://github.com/dimarkov/pybefit/blob/813aeb5ad2c7bbed90ecd7d947bce9fe981c9702/examples/temporal_rev_learn/behavioral_data/pilot/format_behavioral_data.py) 및 [MIT 선언](https://github.com/dimarkov/pybefit/blob/813aeb5ad2c7bbed90ecd7d947bce9fe981c9702/LICENSE.md)을 확인했다. MIT 소프트웨어 선언을 별도 행동 자료 전체에 자동 적용하지 않는다.

아래 행 번호는 원바이트를 보존한 해당 소스 파일 기준이다. MATLAB 링크는 모두 revision 1이며, GitHub 링크는 위 고정 커밋이다. 프로그램을 실행해 과제를 재연하거나 저자의 분석 프로그램을 실행한 것은 아니다.

## 코드와 의미의 행별 판정

| 확인 대상 | 직접 근거 | 판정 및 적용 한계 |
|---|---|---|
| 원 `A`와 가공 `responses` | main formatter 100–101행; pilot 72–73행 | 가공 응답은 원 `A`에서 1을 뺀 값이다. 원 1/2가 가공 0/1, 원 3이 가공 2가 된다. 결측을 선택 0으로 채우지 않는다. |
| 카드 정체성과 화면 위치 | 논문 §4.2.1; [preps.m](https://osf.io/57wr8/files/osfstorage/5d08b68d1ed0c20018939416?revision=1) 32–33행; [opvoltrial.m](https://osf.io/57wr8/files/osfstorage/5d08b68d43bdc30017e87ac0?revision=1) 14–15·38–52·69–79행 | 논문은 자극 위치가 시행 사이에 고정됐다고 명시한다. 공개 코드는 화면 반응 `a_side`를 `random_lr`에 따라 `A`로 보정하므로 `A`는 배치와 구분한 카드 코드다. 현재 기본 배치는 고정이지만 각 2018 원본의 정확한 실행판·자극 그림까지 입증한 것은 아니다. 후속 표현은 ‘source-coded option 0/1’로 제한한다. |
| 카드 0/1의 보상 | opvoltrial 90–97행; main formatter 103–112행 | 카드 선택의 실제 금전 피드백은 해당 시행의 `R`에서 선택한 카드 열을 읽는다. 원 −1/+1을 가공 0/1로 바꾼다. 가공 0은 손실이며 금전 결과 없음과 다르다. |
| 정보 요청 2 | opvoltrial 48–52·76–77·84–85·93–94행 | 원 `A=3`은 도움 요청이다. 그 시행의 실제 금전 결과는 0이고 `C`는 결측이다. 카드 오답으로 바꾸지 않는다. |
| 결과 2/3 | opvoltrial 103·167행; main formatter 112행 | 도움 요청 시 `S`가 가리키는 카드를 강조하고, formatter는 `S+1`을 저장한다. 원 상태 1/2에 대응하는 정보 결과 2/3이며 보상 크기가 아니다. |
| `S`의 뜻 | opvoltrial 82–87·103·167행; 논문 §2·그림 1·§4.3 | 코드에서는 `A`와 비교해 정오를 결정하고 정보 요청 시 정답으로 제시하는 카드다. 논문에서 ‘correct’는 실현된 당첨 여부가 아니라 **보상 확률이 높은 카드**다. 이 결합은 `S`를 유리한 카드의 상태 코드로 해석하는 근거다. 각 원본의 행별 일관성은 아직 이 문서에서 검사하지 않았다. |
| `R`의 뜻 | [runexp.m](https://osf.io/57wr8/files/osfstorage/5d08b68b11bd08001a9f2989?revision=1) 49–55행; opvoltrial 93–97행; 논문 §2·그림 2 뒤 | 조건별로 미리 생성한 카드별 잠재 피드백 표다. 사람에게 실제로 전달된 피드백은 선택 `A`에 해당하는 값뿐이다. `R` 자체를 관측 응답이나 보상 확률 표로 부르지 않는다. 확률적 결과이므로 한 시행의 당첨이 유리한 카드 선택과 같지 않다. |
| `C`의 뜻 | opvoltrial 82–87행; main formatter 115–116행 | 유리한 카드 선택 1, 다른 카드 선택 0, 정보 요청은 결측이라는 코드 정의다. formatter는 `C`를 그대로 복사한다. 정답 카드 번호가 아니다. |
| 무응답과 시간 초과 | [expparams.m](https://osf.io/57wr8/files/osfstorage/5d08b68b3466a6001c900f3b?revision=1) 23행; opvoltrial 33–64·190–216행 | 응답 창은 3초다. 공개 코드는 시간 초과 시 `A`·`C`를 결측으로 남긴다. `R(nt)=NaN`도 포함해 2열 `R`의 첫 열에 선형 인덱스로 쓸 수 있으므로, 원본의 `R` 결측은 별도 보존·검사가 필요하다. 실제 25개에서 이 경로가 발생했다고 주장하지 않는다. |
| 훈련과 본 실험의 시행 수 | expparams 14–18행; runexp 43–47·68–70행; 논문 §4.2.1 | 훈련 40시행, 20시행 뒤 한 차례 반전이다. 본 실험은 1,000시행이다. 훈련을 본 실험 첫 40칸에 합치지 않는다. |
| 시행 번호 | runexp 68–70행; opvoltrial 1·22·183행; main formatter 81·121–134행 | MATLAB `nt`는 1부터 진행한다. CSV 열은 pandas 배열 위치인 0부터다. 순서를 재정렬하지 않을 때 CSV 열 `j`는 원 시행 `j+1`이다. 보류·정보 요청·결측으로 시행을 삭제하면 이 대응이 깨진다. |
| 정보 제공의 가용성 | [instr_opvol.m](https://osf.io/57wr8/files/osfstorage/5d08b8bb1ed0c2001693e76c?revision=1) 27·61행; runexp 61–65행; opvoltrial 17–18·48–52행; [draw_hints.m](https://osf.io/57wr8/files/osfstorage/5d08b69143bdc30017e87ac5?revision=1) 1–6행 | 공개 3선택 과제에서는 매 시행 힌트 버튼을 그리고 도움 키를 허용한다. 논문은 main 50행에 해당하는 3선택 설계와 pilot 24명의 2선택 설계를 구분한다. 정보 요청을 실제 선택하지 않았다는 이유로 ‘제공되지 않았다’고 추론하지 않는다. 원본별 제공 마스크를 새로 복구한 것은 아니다. |
| `offers`와 `random_lr` | preps 32–33행; opvoltrial 14–15·69–79행; main formatter 98행 | `random_lr`는 카드 배치이며 formatter의 `offers`는 그 배치를 변환한 변수다. **정보 요청 offered availability 마스크로 쓰지 않는다.** formatter는 이 `offers`를 세 결과 CSV로 내보내지도 않는다. |
| 정보 제공의 비용 | instr_opvol 61·77행; opvoltrial 93–97행; 논문 §2·그림 1 | 정보 요청에 직접적인 금전 수수료가 없다. 그 시행에는 이득·손실 모두 없으며 카드 선택 대신 정보를 얻는다. 이를 고정된 금전 손실이나 항상 비용이 전혀 없는 행위로 해석하지 않는다. 주관적 정보 가치·기회비용은 별도 모형 가정이다. |
| 지급 금액의 코드 | expparams 6–9행; [instr_payment.m](https://osf.io/57wr8/files/osfstorage/5d08b8bc11bd0800179f84db?revision=1) 4–11행; [setup.m](https://osf.io/57wr8/files/osfstorage/5d08b68b43bdc30016e88ebd?revision=1) 112–116행 | 공개 코드는 ±10센트 표시와 지급액의 하한 0·상한 10유로를 포함한다. 이 설정이 각 2018 원본의 실제 지급과 같았는지는 미확인이다. 이번에 개인별 지급액은 계산하지 않았다. |
| 조건 표시 | main formatter 73–75·123–124행; pilot 45–47·95–96행 | main은 `fnames11+fnames12`가 regular, 뒤 `fnames21+fnames22`가 irregular이다. pilot은 `fnames1`/`fnames2`로 나뉜다. OSF 폴더 이름이나 응답 코드 빈도로 조건을 다시 붙이지 않는다. |

## 반전 일정에서 확인된 범위

논문 §2·그림 2는 두 조건의 **처음 800시행과 마지막 200시행**을 구분한다. regular 조건은 처음에 준규칙적 간격, 마지막에 불규칙적 간격을 경험하고 irregular 조건은 순서가 반대다. 간격 생성은 각각 음이항 분포와 기하 분포로 설명되며, §4.4에서는 평균 약 20시행의 시간 구조를 설명한다. 선택된 일정은 조건마다 한 번 생성해 고정했고, 마지막 200시행은 반대 조건 처음 200시행의 궤적을 복사했다고 명시한다.

공개 코드의 `runexp.m`은 저장된 `regular.S/R` 또는 `irregular.S/R`를 불러와 순서대로 사용한다. **텍스트에는 실제 1,000시행의 모든 반전 번호 목록이 없다.** `states_and_rewards.mat` 파일의 존재·크기 1,823바이트는 공식 목록에서 확인했지만 이번에는 내려받거나 읽지 않았다. 25개 원본의 `S/R` 값도 읽지 않았으므로, 실제 반전 수·경계 번호·조건 내 동일성은 아직 미검사다.

후속 구조 감사에서 연속된 원 상태가 달라진다면 새 상태가 나타난 시행을 경계 후보로 정의할 수 있다. 이는 이 문서의 코드 해석이며 새 계산 규칙은 실행 전에 별도 등록해야 한다. 800→801의 구획 변경은 간격 통계의 변화다. 그 위치에서 반드시 카드 상태 자체도 반전한다고 가정하지 않는다. 또한 과제 일정은 행동 차이가 커지도록 저자 모형으로 선별된 궤적이므로 자연스러운 임의의 반전 환경을 대표한다고 가정하지 않는다.

## 앞서 검증한 25개 원본에 적용할 때

앞선 동결 등록과 원본 재현 결과는 main의 25행과 세 배열이 결측 위치까지 일치함을 확인했다. 해당 범위는 formatter 명단상 `fnames11`·`fnames21`에 해당하고 regular 15행·irregular 10행이다. 이는 기존 결과를 인용한 것이며 이번에 사람 선택을 다시 집계한 값이 아니다. 49행 미검증과 pilot 중복 명단은 그대로 남긴다.

다만 공개 `modifyme.m` 46행의 버전 문자열은 `TEMP_OPVOL_3.0_190611_TUD`이고 OSF revision 1의 게시 시각은 2019-06-18이다. 검증된 명단의 파일명에는 2018년 시각이 들어 있다. `preps.m` 35–36행의 참가자 번호 홀짝 배정은 이 이전 명단의 조건 배정으로 그대로 대체할 수 없다. 공개 `expparams.m` 31행의 피드백 1초와 논문 §4.2.1의 0.5초도 일치하지 않는다. 따라서 **공개 코드가 그 25개 각각의 정확한 실행판이었다는 주장에는 근거가 부족하다.** 이 차이로 원본의 카드 코드·상태 의미가 틀렸다고 입증된 것은 아니다.

논문·formatter·공개 과제 코드가 함께 지지하는 제한적 해석은 다음과 같다. 응답 0/1은 배치 보정된 두 카드 코드, 응답 2는 정보 요청, `S`는 과제가 정답으로 삼는 카드, `R`은 선택별 예정 피드백이다. 실제 25행의 ‘source-coded option 유지·전환’을 기술하려면 먼저 등록된 구조 검사에서 배열 차원·허용값·결측·시행 순서와 유한한 카드 선택의 `C`/`A`/`S` 일관성을 확인하는 것이 적절하다. 공개 코드와의 정확한 버전 연결이 미확인이라는 한계는 그 뒤에도 남는다.

## DrosoLearn에서 새로 정해야 하는 부분

원논문은 DrosoLearn의 커넥톰 모형이 이 과제를 인간과 같은 방식으로 배운다고 입증하지 않았다. 후속 비교의 분모, 정보 요청 사이를 건너뛰어 카드 전환을 셀지, 바로 인접한 시행만 셀지, 결측을 어떻게 제외·보존할지, 800/200 구획을 어떻게 나눌지는 **DrosoLearn의 새 분석 가정**이다. 이번 문서가 그 가정을 대신 등록하지 않는다. 사람과 모형의 차이 검정·정확도·학습 곡선·신경 메커니즘 유사성 주장도 하지 않는다.

## 원문 바이트와 보존 기록

새 작업 폴더 `work/human_task_structure_sources/`에 공식 응답·작은 코드 원바이트와 `source-manifest.json`을 보존했다. 저장소 밖 작업 폴더 기준이며 논문 HTML은 연구 열람 캐시에만 둔다. manifest는 38개 취득 파일 각각의 공식 URL, UTC 취득 시작 시각, HTTP 상태, 바이트, SHA-256을 가진다. GitHub 원문 7개는 고정 커밋의 Git blob SHA-1도 대조했다. OSF 코드 12개는 revision 1 및 공식 목록의 크기·SHA-256과 모두 일치한다. 사람 원본·행동 CSV·EEG·실행 바이너리는 이 감사에서 받지 않았다.

새 [저장소 근거 폴더](../data/human/task_structure_evidence_20261015/README.md)에는 OSF MATLAB 원문 12개, formatter 2개와 MIT 고지, 공식 메타데이터 18개, 새 출처 안내문을 합쳐 34개 파일·130,034바이트를 보존했다. [저장소 manifest](../data/human/task_structure_evidence_20261015/source-manifest.json)의 `files`는 이 번들 파일만 가리키며 `external_review_sources`는 별도 열람 캐시를 구분한다. `semantic_decisions`에 코드 판정과 한계를 직접 기입했다. 논문 HTML은 번들에 없다.

취득 시각 범위는 2026-10-15 03:48:40–03:50:41 UTC다. 작업 폴더의 `source-manifest.json`은 22,783바이트이며 SHA-256은 `804fe3eb97de36f2a8358a29b2ba954d346cdddd30ef295c0c8e180ca9a4386a`다. 저장소 manifest의 SHA-256은 `3e39f3e06bb9cc62569eb851fda88d449dcfbe0e1f4a2fa57efe35fd24ac9fe8`다. 소스별 정확 시각과 해시는 각 manifest가 기준이다. 기존 저장소의 소스·등록·결과는 수정하지 않았다.
