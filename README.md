# 의료기기 DX 배포 실습 예제

공개 K01 데이터로 학습한 로지스틱 회귀를 Streamlit으로 연결한 교육용 예제입니다. 진단 목적으로 사용할 수 없습니다.

- app.py: 실행 파일
- model.json: 사전 학습한 모델·전처리·입력 순서. 앱에서 재학습하지 않습니다.
- example-input.csv: 공개 자료의 범주별 중앙값으로 만든 예시 2건. 실제 개인의 진료기록이 아닙니다.
- requirements.txt: 실제 검증 환경의 패키지
- verification.json: 원래 저장 모델과 이식한 모델의 예측 대조 결과
- Python: 3.11 권장. Community Cloud에서 동일 버전을 선택합니다.

자료 출처: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

원자료: UCI Breast Cancer Wisconsin (Diagnostic), https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic . 원자료의 이용 조건을 확인해 재사용합니다.

수업용 내부 학습 결과이며 외부 기관 검증·임상적 유용성 검증을 수행하지 않았습니다. 예측 확률은 별도로 보정하지 않았습니다.

## 프롬프트로 실행

“이 폴더의 requirements.txt로 환경을 준비하고 Streamlit 앱을 실행해줘. 예시 2건의 결과를 verification.json과 비교해줘.”

## 프롬프트로 게시·배포

“공개할 파일 목록을 확인하고 GitHub에 게시해줘. Streamlit에 입력할 저장소·브랜치·실행 파일·Python 버전을 알려줘.”

원저자: William Wolberg, Olvi Mangasarian, Nick Street, W. Street. UCI 원자료 CC BY 4.0, https://doi.org/10.24432/C5DW2B . 공개 예제 입력은 범주별 중앙값으로 변형했습니다.
