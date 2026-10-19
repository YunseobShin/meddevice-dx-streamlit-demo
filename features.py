import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks,welch,resample_poly
FEATURES=['기록 길이','원 신호 표준편차','진폭 범위','중앙 절대편차','고주파 에너지 비율','저주파 에너지 비율','스펙트럼 엔트로피','검출 박동 수','평균 심박수','박동 간격 평균','박동 간격 변동','박동 간격 변동계수','인접 간격 차이','간격 변화 비율']
def clean(raw,fs=300):
 x=np.asarray(raw,dtype=float).reshape(-1)
 if len(x)<fs*5:raise ValueError('5초 이상의 ECG가 필요합니다.')
 valid=np.isfinite(x)
 if valid.mean()<.98:raise ValueError('결측 구간이 너무 큽니다.')
 if not valid.all():x=np.interp(np.arange(len(x)),np.flatnonzero(valid),x[valid])
 return sosfiltfilt(butter(3,[.5,40],fs=fs,btype='bandpass',output='sos'),x)
def extract(raw,fs=300):
 x=np.asarray(raw,dtype=float).reshape(-1);v=clean(x,fs);std=np.std(v)+1e-8;z=(v-np.mean(v))/std
 energy=np.convolve(np.diff(z,prepend=z[0])**2,np.ones(30)/30,'same')
 peaks,_=find_peaks(energy,distance=int(.25*fs),prominence=max(np.percentile(energy,90)*.35,1e-5))
 rr=np.diff(peaks)/fs;rr=rr[(rr>.25)&(rr<2.5)]
 f,p=welch(z,fs,nperseg=min(len(z),1024));p=p+1e-12;prob=p/p.sum()
 m=rr.mean() if len(rr) else np.nan;s=rr.std() if len(rr) else np.nan;diff=np.diff(rr)
 return np.array([len(x)/fs,np.nanstd(x),np.nanpercentile(x,99)-np.nanpercentile(x,1),np.nanmedian(abs(x-np.nanmedian(x))),p[f>20].sum()/p.sum(),p[f<5].sum()/p.sum(),-(prob*np.log(prob)).sum(),len(peaks),60/m if m>0 else np.nan,m,s,s/m if m>0 else np.nan,np.sqrt(np.mean(diff**2)) if len(diff) else np.nan,np.mean(abs(diff)>.05) if len(diff) else np.nan])
def cnn_input(raw,fs=300):
 v=clean(raw,fs);v=(v-v.mean())/(v.std()+1e-8);v=resample_poly(v,100,fs).astype('float32')[:6100]
 out=np.zeros(6100,dtype='float32');out[:len(v)]=v
 return out,len(v)
