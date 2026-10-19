# UCI 표 데이터의 심장질환 분류 실습

공통 입력: 나이·안정 시 혈압·콜레스테롤·최대 심박수·흉통 유형.
저장한 랜덤포레스트200개, 깊이2, 최소 말단2건. 학습181·검증61·시험61건.
공개 데이터 예제, 직접 입력, 한 행 CSV로 질환 유무를 예측합니다.

## Codex에 요청

이 폴더의 패키지를 설치하고 Streamlit 앱을 실행해줘. 주소를 열고 공개 예제와 직접 입력에서 저장 모델의 예측값이 같은지 확인해줘.

## 배포

Codex에 Git·GitHub CLI 설치와 계정 연결을 요청합니다. 공개할 파일 목록을 확인하고 GitHub에 게시하도록 요청합니다. Streamlit Community Cloud에서 이 저장소의 main 브랜치, app.py를 지정합니다. Python3.11을 사용합니다.

입력 하나를 학습 중앙값·최빈값으로 바꾼 민감도 그림은 SHAP·인과효과와 구분합니다. 전체 모델 비교의 검증 선택은LightGBM이며, 이앱은확률평균을시연하는랜덤포레스트입니다. 임상진단용이아닙니다.

데이터: UCI Heart Disease (Janosi, Steinbrunn, Pfisterer, Detrano,1989), https://archive.ics.uci.edu/dataset/45/heart+disease , CC BY4.0. 원자료의5개 열을 선택하고 질환유무를 파생했습니다. verification.json은303건의 저장모델 예측 대조 결과입니다.
