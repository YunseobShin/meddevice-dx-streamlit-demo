from pathlib import Path
import json,numpy as np
MODEL=json.loads((Path(__file__).parent/'rhythm-model.json').read_text(encoding='utf-8'))
def predict(x):
 x=np.atleast_2d(x).astype(float);x=np.where(np.isnan(x),MODEL['median'],x);z=((x-MODEL['mean'])/MODEL['scale']).astype('float32');result=np.zeros((len(z),4))
 for t in MODEL['trees']:
  for i,row in enumerate(z):
   node=0
   while t['left'][node]!=-1:node=t['left'][node] if float(row[t['feature'][node]])<=t['threshold'][node] else t['right'][node]
   values=np.array(t['value'][node]);result[i]+=values/values.sum()
 return result/len(MODEL['trees'])
