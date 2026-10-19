from pathlib import Path
import json,numpy as np,pandas as pd,streamlit as st
from features import extract,FEATURES
from predict import predict,MODEL
ROOT=Path(__file__).parent;labels=MODEL['labels'];examples=json.loads((ROOT/'examples.json').read_text(encoding='utf-8'))
st.set_page_config(page_title='ECG 부정맥 판독 실습',layout='wide')
st.title('ECG 파일로 부정맥 판독')
st.caption('PhysioNet/CinC 2017 · 수업용 모델, 임상 진단용 아님')
st.write('ECG 한 기록을 넣으면 정상 리듬·심방세동·기타 리듬·판독 곤란 중 하나를 예측합니다.')
mode=st.radio('입력 방법',['공개 ECG 예제','CSV 업로드'],horizontal=True);data=None;truth=None
if mode=='공개 ECG 예제':
 choice=st.selectbox('ECG 선택',list(range(4)),format_func=lambda c:f'{labels[c]} 예제 · {examples[c]["record"]}',index=1)
 data=pd.read_csv(ROOT/f'example-{choice}.csv');truth=examples[choice]['truth']
 st.caption(f'원자료 판독 정답: {truth} · 기록 {examples[choice]["record"]}')
else:
 uploaded=st.file_uploader('300Hz, mV 단위의 ECG CSV (ecg_mV 열 하나)',type=['csv'])
 st.download_button('입력 형식 예제 받기',(ROOT/'example-1.csv').read_bytes(),'ecg-example.csv','text/csv')
 if uploaded:
  try:data=pd.read_csv(uploaded)
  except Exception:st.error('CSV를 읽을 수 없습니다. 파일 형식을 확인해 주세요.');st.stop()
if data is not None:
 if list(data.columns)!=['ecg_mV']:st.error('ecg_mV 열 하나만 있는 파일을 사용해 주세요.');st.stop()
 try:raw=pd.to_numeric(data.ecg_mV,errors='raise').to_numpy(dtype=float)
 except Exception:st.error('ECG 열에는 수치가 필요합니다.');st.stop()
 if not 2700<=len(raw)<=18300:st.error('이 모델은 300Hz, 9～61초 기록을 사용합니다. 입력 조건을 확인해 주세요.');st.stop()
 try:x=extract(raw)
 except ValueError as e:st.error(str(e));st.stop()
 p=predict(x)[0];c=int(p.argmax());st.subheader('판독 결과: '+labels[c])
 st.write(f'가장 높은 모델 점수: {p[c]:.3f}'+(f' · 원자료 정답과 {"일치" if labels[c]==truth else "불일치"}' if truth else ''))
 st.dataframe(pd.DataFrame({'판독 범주':labels,'모델 점수':np.round(p,4)}),hide_index=True,width='stretch')
 st.caption('가장 높은 점수의 범주를 선택합니다. 점수는 보정된 진단 확률이 아닙니다.')
 st.subheader('입력 ECG');st.line_chart(pd.DataFrame({'시간(초)':np.arange(len(raw))/300,'ECG(mV)':raw}),x='시간(초)',y='ECG(mV)',color='#2457d6',height=220)
 with st.expander('예측에 영향을 준 특징 확인'):
  altered=np.repeat(x[None],len(x),axis=0)
  for j in range(len(x)):altered[j,j]=MODEL['mean'][j]
  delta=p[c]-predict(altered)[:,c];idx=np.argsort(abs(delta))[-5:];st.bar_chart(pd.Series(delta[idx],index=[FEATURES[j] for j in idx]),horizontal=True,color='#2457d6')
  st.caption('특징 하나를 학습 평균으로 바꿨을 때 선택한 클래스의 점수 감소. SHAP이나 질환의 원인을 뜻하지 않습니다. 특징 사이 상관의 영향을 받을 수 있습니다.')
 st.download_button('예측 결과 받기',pd.DataFrame({'판독 범주':labels,'모델 점수':p}).to_csv(index=False).encode('utf-8-sig'),'prediction.csv','text/csv')
with st.expander('모델과 검증 범위'):
 st.write('랜덤 포레스트, 트리120개, 깊이16. 기록별 특징14개. 학습1,439·검증480·시험481건, 시험 macro-F1 0.659. 기록·동일 신호 중복을 막았지만 환자 ID가 공개되지 않아 환자 독립성을 확인하지 못했습니다.')
 st.write('기타 리듬은 특정 부정맥 하나가 아닙니다. 심실빈맥 전용 라벨은 없습니다. 입력을 별도 저장하지 않습니다.')
 st.link_button('공식 데이터와 판독 기준',MODEL['source'])
st.caption('배포 확인: ECG 부정맥 판독 모델 반영 완료 · 같은 입력의 예측값을 다시 확인합니다.')
