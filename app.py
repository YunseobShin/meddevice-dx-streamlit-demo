from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).parent
MODEL=json.loads((ROOT/'model.json').read_text(encoding='utf-8'))
FEATURES=MODEL['features']
st.set_page_config(page_title='의료기기 DX · 예측 시제품',layout='wide')
st.title('저장 모델로 실행하는 예측 시제품')
st.caption('연구용 시제품, 진단 목적 사용 불가 · 공개 K01 학습 예제')
st.write('예시 또는 CSV를 입력하면 저장한 로지스틱 회귀의 예측과 특징 기여도를 확인할 수 있습니다.')
mode=st.radio('입력 방법',['예시 데이터','CSV 업로드'],horizontal=True)
data=None
if mode=='예시 데이터':
    choices=pd.read_csv(ROOT/'example-input.csv')
    selected=st.selectbox('예시 선택',['예시 A','예시 B'])
    data=choices.iloc[[0 if selected=='예시 A' else 1]].copy()
    st.caption('예시는 공개 자료의 범주별 중앙값으로 만든 입력입니다. 개인의 진료기록이 아닙니다.')
else:
    uploaded=st.file_uploader('입력 특징 30열이 있는 CSV',type=['csv'])
    st.download_button('예시 입력 CSV 받기',(ROOT/'example-input.csv').read_bytes(),'example-input.csv','text/csv')
    if uploaded:
        try:data=pd.read_csv(uploaded)
        except Exception:st.error('CSV를 읽을 수 없습니다. 구분자와 문자 인코딩을 확인해 주세요.')
if data is not None:
    missing=[name for name in FEATURES if name not in data.columns]
    if missing:
        st.error('필수 입력 열이 없습니다: '+', '.join(missing))
        st.stop()
    if data.empty:st.warning('입력 행이 없습니다.');st.stop()
    try:values=data[FEATURES].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    except (ValueError,TypeError):st.error('특징 열에는 수치만 입력해 주세요.');st.stop()
    if np.isinf(values).any():st.error('무한대 값은 입력할 수 없습니다.');st.stop()
    values=np.where(np.isnan(values),np.array(MODEL['median']),values)
    z=(values-np.array(MODEL['mean']))/np.array(MODEL['scale'])
    contributions=z*np.array(MODEL['coef'])
    logits=contributions.sum(axis=1)+MODEL['intercept']
    probability=1/(1+np.exp(-np.clip(logits,-700,700)))
    output=pd.DataFrame({'예측 범주':np.where(probability>=MODEL['threshold'],'M','B'),'M 예측 확률':probability,'임계값':MODEL['threshold']})
    st.subheader('예측 결과')
    st.dataframe(output,hide_index=True,width='stretch')
    st.caption('B·M은 학습 자료의 범주입니다. 임계값 0.5, 확률은 별도로 보정하지 않았습니다.')
    st.subheader('첫 번째 입력의 특징 기여도')
    c=pd.Series(contributions[0],index=FEATURES)
    top=c.loc[c.abs().nlargest(6).index].sort_values()
    st.bar_chart(top,horizontal=True,color='#2457d6')
    st.caption('표준화 특징 × 학습 계수. 양수는 M의 로그 오즈를 높이고 음수는 낮춥니다. SHAP 그림이나 질환의 원인으로 해석하지 않습니다.')
    st.download_button('예측 CSV 받기',output.to_csv(index=False).encode('utf-8-sig'),'predictions.csv','text/csv')
with st.expander('모델·입력·검증 정보'):
    st.write('모델: 로지스틱 회귀 C=0.1. 학습 자료에서 추정한 결측 대체·표준화와 30개 입력 순서를 저장해 사용합니다.')
    st.write('수업용 내부 분할 결과이며 외부 기관·임상 환경에서 검증하지 않았습니다. 앱은 입력을 별도 저장하지 않습니다.')
    st.write('모델 버전: '+MODEL['version'])
    st.json({'inputFeatures':FEATURES,'source':MODEL['source']})
st.caption('배포 확인: GitHub 업데이트 반영 완료 · 같은 입력의 예측값을 다시 확인합니다.')
