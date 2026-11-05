# -*- coding: utf-8 -*-
"""Cohen's kappa between rater 1 and rater 2. Usage:
python M19_Kappa_Analyze.py --form M19_Second_Rater_Blind_Review_Form.xlsx --key M19_Second_Rater_Private_Key.csv
Primary: binary (Correct vs not-Correct; Partial counted as Incorrect, pre-specified). Secondary: 3-class, error-type among both-not-correct,
and by source (V2F / Real). Bootstrap 95% CI (2000 reps, seed 20261103)."""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
ap=argparse.ArgumentParser();ap.add_argument('--form',default='M19_Second_Rater_Blind_Review_Form.xlsx');ap.add_argument('--key',default='M19_Second_Rater_Private_Key.csv')
a=ap.parse_args()
f=pd.read_excel(a.form,sheet_name='Review');k=pd.read_csv(a.key,encoding='utf-8-sig')
m=k.merge(f[['review_id','label','error_type_if_not_correct']],on='review_id')
assert m.label.notna().all(), f'{m.label.isna().sum()} unlabeled rows'
m['b1']=(m.first_label=='Correct').astype(int);m['b2']=(m.label=='Correct').astype(int)
def kap(x,y):
    return cohen_kappa_score(x,y) if len(set(x))>1 or len(set(y))>1 else float('nan')
def ci(d,col1,col2):
    r=np.random.default_rng(20261103);v=[]
    for _ in range(2000):
        s=d.sample(len(d),replace=True,random_state=int(r.integers(1e9)));v.append(kap(s[col1],s[col2]))
    return np.nanpercentile(v,[2.5,97.5])
def rep(name,d,c1,c2):
    ag=(d[c1]==d[c2]).mean();kk=kap(d[c1],d[c2]);lo,hi=ci(d,c1,c2)
    print(f'{name}: n={len(d)} agreement={ag:.3f} kappa={kk:.3f} [{lo:.3f},{hi:.3f}]')
rep('Binary, all',m,'b1','b2')
for s,d in m.groupby('source'): rep(f'Binary, {s}',d,'b1','b2')
m['t1']=m.first_label.replace({'Incorrect':'Incorrect'});rep('3-class (Correct/Partial/Incorrect), all',m,'first_label','label')
both=m[(m.b1==0)&(m.b2==0)].copy()
if len(both):
    t1=both.first_type.fillna('').astype(str).str.lower().str.split().str[0];t2=both.error_type_if_not_correct.fillna('').astype(str).str.lower()
    both['t1']=t1;both['t2']=t2;print(f'Error type among both-not-correct: n={len(both)} agreement={(both.t1.str[:5]==both.t2.str[:5]).mean():.3f}')
print('Confusion (rows=rater1, cols=rater2):');print(pd.crosstab(m.first_label,m.label))
m.to_csv('M19_Kappa_Merged.csv',index=False,encoding='utf-8-sig')
print('Disagreements:');print(m[m.b1!=m.b2][['review_id','case_id','source','first_label','label']].to_string(index=False))
