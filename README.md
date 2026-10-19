# ECG 부정맥 판독 시제품

PhysioNet/CinC 2017의 ECG 한 기록을 정상 리듬, 심방세동, 기타 리듬, 판독 곤란으로 분류하는 수업용 앱입니다.

입력은 300Hz, mV 단위의 ecg_mV 열이 있는 CSV입니다. 공개 기록 네 건을 예제로 포함했습니다. 원문 출처와 기록 번호는 앱에 표시합니다.

Codex에 다음과 같이 요청합니다.

“이 폴더의 의존성을 설치하고 Streamlit 앱을 실행해서 브라우저로 열어줘. 예제 네 건의 예측과 저장 모델 결과가 일치하는지 확인해줘.”

GitHub 게시도 Codex에 요청합니다.

“공개 가능한 파일을 확인하고 GitHub 저장소에 게시해줘. Streamlit 배포에 쓸 저장소, 브랜치, 실행 파일, Python 버전을 알려줘.”

모델은 랜덤 포레스트 120개 트리, 깊이16입니다. 원래 모델의 전처리와 트리를 JSON으로 내보내고 시험481건에서 예측 일치를 검증했습니다. 2,400건 교육용 부분집합의 내부 평가이며 환자 식별자가 공개되지 않아 환자 독립성은 확인할 수 없습니다. 임상 진단용 모델이 아닙니다.

공식 데이터: https://physionet.org/content/challenge-2017/1.0.0/
Clifford et al., AF Classification from a Short Single Lead ECG Recording, PhysioNet/CinC Challenge 2017. DOI: 10.22489/CinC.2017.065-469.
