from pathlib import Path
import pandas as pd
import streamlit as st
from predict import predict, MODEL
ROOT=Path(__file__).parent
st.set_page_config(page_title='표 데이터의 심장질환 분류 실습',layout='wide')
st.title('환자별 표 데이터로 심장질환 분류')
st.caption('UCI Heart Disease · 수업용 랜덤포레스트, 임상 진단용 아님')
st.write('1일차에 학습한 같은 입력 5개와 저장 모델로 질환 있음·없음을 예측합니다.')
examples=pd.read_csv(ROOT/'tabular-examples.csv')
mode=st.radio('입력 방법',['공개 데이터 예제','직접 입력','CSV 업로드'],horizontal=True)
truth=None;row=None
if mode=='공개 데이터 예제':
 index=st.selectbox('환자별 검사 기록',list(range(len(examples))),format_func=lambda i:f'원자료 {int(examples.iloc[i].row_id)+1}번 · '+('질환 있음' if examples.iloc[i].num>0 else '질환 없음'))
 row=examples.iloc[[index]][MODEL['features']];truth=int(examples.iloc[index].num>0)
elif mode=='직접 입력':
 a,b=st.columns(2)
 with a:
  age=st.number_input('나이(세)',18,110,63);bp=st.number_input('안정 시 혈압(mmHg)',50,250,130);chol=st.number_input('콜레스테롤(mg/dL)',50,650,330)
 with b:
  hr=st.number_input('최대 심박수(회/분)',40,240,132);cp=st.selectbox('흉통 유형',[1,2,3,4],index=3,format_func=lambda x:{1:'전형적 협심증',2:'비전형적 협심증',3:'비협심증성 통증',4:'무증상'}[x])
 row=pd.DataFrame([[age,bp,chol,hr,cp]],columns=MODEL['features'])
else:
 st.download_button('입력 CSV 예제 받기',examples[MODEL['features']].head(1).to_csv(index=False).encode('utf-8-sig'),'heart-example.csv','text/csv')
 file=st.file_uploader('입력 5개 열이 있는 CSV, 한 행씩 예측',type=['csv'])
 if file:
  try:
   data=pd.read_csv(file)
   if not set(MODEL['features']).issubset(data.columns):raise ValueError('age, trestbps, chol, thalach, cp 열이 필요합니다.')
   if len(data)!=1:raise ValueError('이 시연 앱에는 한 행만 입력해 주세요.')
   row=data[MODEL['features']].apply(pd.to_numeric,errors='raise')
  except Exception as e:st.error(str(e));st.stop()
if row is not None:
 st.dataframe(row.rename(columns=dict(zip(MODEL['features'],['나이','안정 시 혈압','콜레스테롤','최대 심박수','흉통 유형']))),hide_index=True,width='stretch')
 try:p=float(predict(row)[0])
 except ValueError as e:st.error(str(e));st.stop()
 pred=int(p>=.5);st.subheader('예측: '+MODEL['labels'][pred]);st.write(f'질환 있음 점수: {p:.4f} · 판정 기준: 0.5')
 if truth is not None:st.write(f'원자료 정답: {MODEL["labels"][truth]} · 예측과 '+('일치' if pred==truth else '불일치'))
 st.caption('점수는 200개 트리의 클래스 확률 평균입니다. 보정된 개인의 질환 발생 위험을 뜻하지 않습니다.')
 with st.expander('이 예측에 영향을 준 입력',expanded=True):
  base=MODEL['reference'];rows=pd.concat([row]*5,ignore_index=True)
  for j,col in enumerate(MODEL['features']):rows.loc[j,col]=base[col]
  delta=p-predict(rows)
  st.bar_chart(pd.Series(delta,index=['나이','안정 시 혈압','콜레스테롤','최대 심박수','흉통 유형']),horizontal=True,color='#2457d6')
  st.caption('입력 하나를 학습 자료의 중앙값 또는 최빈값으로 바꿨을 때 원래 점수와의 차이입니다. SHAP·인과효과와 구분하며, 다른 입력과의 관계에 영향을 받을 수 있습니다.')
 st.download_button('예측 결과 받기',pd.DataFrame({'예측':[MODEL['labels'][pred]],'질환 있음 점수':[p]}).to_csv(index=False).encode('utf-8-sig'),'prediction.csv','text/csv')
with st.expander('모델과 검증 범위'):
 st.write('UCI Cleveland 303건, 공통 입력 5개. 학습 181·검증 61·시험 61건. 랜덤포레스트 200개, 최대 깊이 2, 최소 말단 표본 2건. 검증 macro-F1 0.769, 시험 0.716.')
 st.write('검증 점수로 고른 전체 비교 모델은 LightGBM입니다. 이 배포 시연은 확률 평균을 확인하기 위한 랜덤포레스트를 사용합니다. 임상·외부 검증과 확률 보정은 수행하지 않았습니다.')
 st.link_button('공식 데이터와 변수 설명',MODEL['source'])
st.caption('배포 확인: UCI 표 데이터와 랜덤포레스트 반영 완료 · 저장 모델과 같은 입력의 예측을 대조합니다.')
