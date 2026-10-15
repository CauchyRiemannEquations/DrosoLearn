# 인간 기대·선택 코드의 추가 출처 확인

작성일: 2026-10-15 (Asia/Seoul). 기존 대응 감사의 기대 코드 불일치와 MAT action 의미를 확인하기 위한 제한된 출처 조사다. 새 인간 시행별 재매칭, 코드 순열 탐색, 학습 성과 분석은 수행하지 않았다.

쉽게 말하면, 서로 연결된 것으로 보이는 두 기록에서 숫자가 다를 때 **원 실험의 버튼 설명이나 변환 코드를 찾아 숫자의 뜻을 먼저 확인하는 단계**다. 기록이 다르다는 이유만으로 어느 쪽이 오답인지 정하거나, 더 잘 맞는 숫자 변환을 채택하지 않는다.

## 확인한 범위

[OpenNeuro ds004295 버전 1.0.0](https://openneuro.org/datasets/ds004295/versions/1.0.0)의 고정 커밋 `c7c5d3a164425af3e65cc908ef0d41f01725984c` 전체 Git tree를 확인했다. [고정 tree API](https://api.github.com/repos/OpenNeuroDatasets/ds004295/git/trees/c7c5d3a164425af3e65cc908ef0d41f01725984c?recursive=1)는 `truncated=false`이며 파일 경로에 해당하는 blob은 183개다. 웹 화면의 파일 수와 같은 집계라고 가정하지 않는다.

183개 경로의 Git 객체 SHA-1, 모드, 객체 크기를 새 `data/human/stolz_2022_semantics_evidence/repository-inventory.json`에 보존했다. 97개는 Git-annex 포인터다. 이 작은 포인터 텍스트의 SHA-256과 Git blob SHA-1을 검증하고, 포인터가 선언한 원 payload의 크기·SHA-256을 기록했다. **이번 조사에서 그 97개 원 payload를 내려받았다는 뜻은 아니다.** 각 항목의 `payload_downloaded_for_this_audit=false`와 전체 0건 표시가 이를 구분한다. 파일 해시와 출처는 같은 폴더의 `source-manifest.json`에 있다.

포인터는 큰 파일의 실제 내용 대신 위치와 해시를 가리키는 짧은 텍스트다. 포인터 크기를 실제 MAT·EEG·그림의 크기로 보고하지 않는다. payload 해시는 포인터가 선언한 값이며, 이번에 받지 않은 payload의 실제 바이트를 새로 검산한 결과로 부르지 않는다.

## 과제 실행·원 로그를 찾은 결과

고정 tree에서 MATLAB 파일은 기존에 검토한 5개뿐이다. Presentation 과제 실행 파일의 일반적인 확장자 `.sce`, `.pcl`, `.exp`, 원 `.log`, 별도 변환 `.py` 파일은 이 tree에 없다. 두 텍스트 경로 `code/Data/eeg/sdfsdf.txt`, `code/eeg/sdfsdf.txt`의 annex 선언 크기는 모두 0바이트다.

이 결론은 **이번 고정 공개 저장소의 경로 목록에 한정**한다. 저자가 해당 코드를 만들지 않았다거나, 다른 곳에도 존재하지 않는다는 뜻이 아니다. 기존 5개 MATLAB 분석 코드의 해석은 `docs/human-trial-linkage-source-review.md`에 보존되어 있으며 이번에 수정하지 않았다.

추가로 이름이 확인된 작은 MAT 후보는 다음과 같다. 이름은 내용의 증거가 아니므로 행·변수·코드 뜻을 새로 확인했다고 보고하지 않는다.

| 원 경로 | 포인터가 선언한 payload 크기 | 이번 처리 |
|---|---:|---|
| `code/Data/RevEx1_BadEEGTrials.mat` | 3,273바이트 | 포인터만 확인, MAT 미다운로드·미해독 |
| `code/Data/RevEx1_FeedbackMarkerbased.mat` | 2,638바이트 | 포인터만 확인, MAT 미다운로드·미해독 |
| `code/Data/RevEx1_tInterval_Select_Feedback.mat` | 172,426바이트 | 포인터만 확인, MAT 미다운로드·미해독 |

그 밖의 `code/Saves/` MAT는 이름상 분석 산출물 후보로 목록에 남겼다. 내용을 읽거나 원 자료를 대신하는 근거로 사용하지 않았다. 26개 EEG `.set`의 선언 크기는 약 20.2–42.8 MB이며, 큰 EEG 파일과 `.fdt` 신호 payload는 내려받지 않았다. 해당 파일 안에 추가 단서가 있는지는 이번 조사로 확인하지 않았다.

## 논문과 보충자료

[원 논문 Methods 2.2, 2.5, 2.6](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14235)은 Presentation 18.2 사용, 시행마다 도형 좌우 무작위 배치, 기대 보고 0–3, 도형별 선택을 사용하는 모형을 설명한다. 이 설명만으로 MAT action 1/2를 특정 도형이나 화면 좌우에 직접 대응시킬 수는 없다. 논문의 과제 설계 설명과 실제 로그→MAT 변환 규칙을 구분한다.

논문에 연결된 `psyp14235-sup-0001-AppendixS1.docx`는 웹 표시상 27.8 KB지만 정상 공개 다운로드 요청에서 HTTP 403이었다. 실제 파일을 받지 못했으므로 바이트 크기·SHA-256·표 S7의 내용은 확인하지 않았다. 접근 차단을 우회하거나 저자에게 연락하지 않았다.

[저자 소속기관의 공개 기록](https://research.gold.ac.uk/id/eprint/33034/)도 정상 접근으로 확인했다. 확보한 HTML의 파일 링크에서는 논문 PDF를 찾았고 별도 부록 DOCX 링크는 확보하지 못했다. 이를 기관의 모든 비공개·과거 자료에 대한 결론으로 확대하지 않는다. 기관 HTML과 접근 결과는 외부 scratch에만 보존했으며, Wiley 논문·보충자료 본문을 저장소에 재배포하지 않았다.

## 현재 판정

이번 제한 탐색에서는 기대 코드 불일치를 설명하는 새로운 버튼 대응표·원 로그·변환 코드를 확보하지 못했다. MAT action의 도형/좌우 대응과 원 시행 번호·도형 위치 복원도 해결되지 않았다. 기존 `+3` 진단을 성공으로 바꾸거나 새로운 변환을 채택할 근거가 추가된 것은 아니다.

후속 진단이 필요하면 허용할 비교표, 분모, 코드 표현과 해석 범위를 별도 프로토콜에 먼저 고정한다. 수치 대응의 불일치는 자료 의미에 관한 결과이며 인간의 학습 실패·확신·수학 이해의 증거가 아니다.
