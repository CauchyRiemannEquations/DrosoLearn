# 원자료 대응을 위한 공식 OSF 메타데이터

공식 OSF API의 공개 원 응답 5개를 바이트 그대로 보존했다. 요청 URL, UTC 접근 시각, HTTP 상태, 크기와 직접 계산한 응답 SHA-256은 `source-manifest.json`에 있다. Marković, Reiter & Kiebel (2022), DOI 10.3389/fnbeh.2022.962494, OSF h526v의 데이터 component 75jrf가 원 출처다. 해당 component의 명시 CC BY 4.0 관계와 공식 라이선스 객체도 함께 보존했다. 라이선스: https://creativecommons.org/licenses/by/4.0/legalcode. 원 응답은 수정하지 않았으며 이 설명과 manifest는 새 DrosoLearn 문서다.

이 파일들은 원래 공개된 **파일 목록·권한 메타데이터**다. 실제 참가자 행동 MAT나 가공 CSV는 포함하지 않는다. 원래의 공개된 코드·숫자 시각으로 된 파일명은 목록 원문에 남아 있다. 제한된 문자패턴 점검에서 개인마다 달라지는 영문 이름 토큰은 관찰하지 않았으며, 실제 사람과 코드의 대응은 조사하지 않았다. 개별 값 대조 보고서에는 원파일명을 출력하지 않는다.

시간에 따라 바뀌는 live API 응답을 다시 받아 동일 해시를 기대하는 대신, 등록기의 `--fetch`는 이 보존본을 새 외부 cache로 복사해 해시를 검증한다. 원본 MAT와 저자의 변환 코드는 공식 revision 또는 Git 커밋 URL에서 받아 외부 cache에만 둔다. OSF의 명시 라이선스를 별도 GitHub 가공 CSV에 자동 적용하지 않는다. 사람의 성적·학습 곡선·모델과의 유사성은 여기서 계산하지 않았다.
