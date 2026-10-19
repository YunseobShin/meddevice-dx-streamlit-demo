from pathlib import Path
import json,numpy as np
MODEL=json.loads((Path(__file__).parent/'tabular-model.json').read_text(encoding='utf-8'))
def predict(rows):
 x=rows[MODEL['features']].to_numpy(dtype=float)
 if np.isinf(x).any():raise ValueError('무한대 값은 입력할 수 없습니다.')
 numeric=np.where(np.isnan(x[:,:4]),np.array(MODEL['numeric_medians']),x[:,:4]);cp=np.where(np.isnan(x[:,4]),MODEL['cp_mode'],x[:,4])
 if not np.isin(cp,MODEL['cp_categories']).all():raise ValueError('흉통 유형은 1, 2, 3, 4 중 하나여야 합니다.')
 z=np.column_stack([numeric,*[cp==c for c in MODEL['cp_categories']]])
 out=[]
 for row in z:
  p=[]
  for tree in MODEL['trees']:
   node=0
   while tree['left'][node]>=0:node=tree['left'][node] if row[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
   p.append(tree['positive'][node])
  out.append(np.mean(p))
 return np.array(out)
