#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare blinded reference and evidence judge scores against confirmed labels.
AUC descriptive only; 5 question clusters; no tuned threshold advertised as held-out.
"""
import argparse,csv,collections,json,math
from pathlib import Path

def read(path):
 with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def auc(errors,scores):
 pos=[s for y,s in zip(errors,scores) if y];neg=[s for y,s in zip(errors,scores) if not y]
 if not pos or not neg:return None
 return sum((a<b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def metrics(errors,scores,t):
 pred=[s<t for s in scores];tp=sum(p and y for p,y in zip(pred,errors));fp=sum(p and not y for p,y in zip(pred,errors));tn=sum(not p and not y for p,y in zip(pred,errors));fn=sum(not p and y for p,y in zip(pred,errors))
 pr=tp/(tp+fp) if tp+fp else None;rec=tp/(tp+fn) if tp+fn else None
 return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision_error':pr,'recall_error':rec,'f1_error':2*pr*rec/(pr+rec) if pr is not None and rec is not None and pr+rec else None}
def main():
 p=argparse.ArgumentParser()
 root=Path(__file__).resolve().parent
 p.add_argument('--reference',type=Path,default=root/'M19_Controlled_Reference_Scores.csv')
 p.add_argument('--evidence',type=Path,default=root/'M19_Controlled_Evidence_Scores.csv')
 p.add_argument('--labels',type=Path,default=root/'M19_Controlled_Expert_Confirmed_Labels.csv')
 p.add_argument('--output',type=Path,default=root/'M19_Controlled_Judge_Comparison.csv')
 p.add_argument('--summary',type=Path,default=root/'M19_Controlled_Judge_Comparison_Summary.json')
 p.add_argument('--threshold',type=float,default=None,help='Only use a threshold preselected without consulting these cases')
 a=p.parse_args()
 datasets=[read(x) for x in (a.reference,a.evidence,a.labels)]
 indexes=[]
 for d in datasets:
  ids=[x['case_id'] for x in d]
  if len(ids)!=len(set(ids)):p.error('Duplicate case_id')
  indexes.append({x['case_id']:x for x in d})
 if set(indexes[0])!=set(indexes[1]) or set(indexes[0])!=set(indexes[2]):p.error('Mismatched case IDs')
 out=[]
 for case in sorted(indexes[0]):
  r,e,l=[d[case] for d in indexes]
  if r['benchmark_id']!=e['benchmark_id'] or e['benchmark_id']!=l['benchmark_id']:p.error('Question mismatch: '+case)
  label=int(l['expert_label_correct_1_error_0'])
  if label not in (0,1):p.error('Invalid expert label')
  out.append({'case_id':case,'benchmark_id':l['benchmark_id'],'error_type':l['error_type'],
              'expert_label_correct_1_error_0':label,
              'reference_agreement':float(r['v2_reference_agreement']),
              'evidence_agreement':float(e['evidence_judge_raw']),
              'evidence_numeric_warning':e['evidence_numeric_warning'],
              'evidence_best_source_id':e['evidence_best_source_id']})
 with a.output.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 y=[1-x['expert_label_correct_1_error_0'] for x in out]
 summary={'n':len(out),'question_clusters':len(set(x['benchmark_id'] for x in out)),
          'confirmed_correct':y.count(0),'confirmed_error':y.count(1),'threshold_status':'not_provided' if a.threshold is None else 'user_supplied_not_independently_verified',
          'limitations':'Synthetic edits; only five question clusters; scores are agreement proxies; label confirmation is user-reported.'}
 for label,col in [('reference','reference_agreement'),('evidence','evidence_agreement')]:
  scores=[x[col] for x in out]
  result={'roc_auc_descriptive':auc(y,scores),'mean_correct':sum(s for s,z in zip(scores,y) if not z)/y.count(0),
          'mean_error':sum(s for s,z in zip(scores,y) if z)/y.count(1),
          'by_error_type':{k:sum(x[col] for x in out if x['error_type']==k)/sum(x['error_type']==k for x in out) for k in sorted(set(x['error_type'] for x in out))}}
  if a.threshold is not None:result['threshold_metrics_descriptive_only']=metrics(y,scores,a.threshold)
  summary[label]=result
 # Within-question pairwise difference to the correct response for each error type
 for label,col in [('reference','reference_agreement'),('evidence','evidence_agreement')]:
  pairs=[]
  for bid in sorted(set(x['benchmark_id'] for x in out)):
   rows=[x for x in out if x['benchmark_id']==bid];correct=[x for x in rows if x['expert_label_correct_1_error_0']==1]
   if len(correct)!=1:raise ValueError('Expected one correct per question: '+bid)
   for x in rows:
    if x['expert_label_correct_1_error_0']==0:pairs.append(correct[0][col]-x[col])
  summary[label]['paired_score_drops_positive']=sum(d>0 for d in pairs)
  summary[label]['paired_score_drops_total']=len(pairs)
 a.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 print('Saved',a.output,'and',a.summary)
if __name__=='__main__':main()
