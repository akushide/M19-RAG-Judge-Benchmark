#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blinded controlled-case reference evaluation. No API/Chroma calls.

Stage 1: score 30 cases using the exact original LocalJudging_v2.evaluate().
Stage 2: OPTIONAL analyse expert-confirmed labels, never mutation labels alone.
This is REFERENCE judge only; evidence judge requires separate fixed evidence and a fresh run.
"""
from __future__ import annotations
import argparse,importlib.util,json
from pathlib import Path
import pandas as pd
import numpy as np

def load_judge(path):
    spec=importlib.util.spec_from_file_location('original_local_judge',str(path))
    if spec is None or spec.loader is None: raise RuntimeError('Cannot load judge: '+str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def main():
    p=argparse.ArgumentParser(description='M19 controlled test: blinded reference judge')
    root=Path(__file__).resolve().parent
    p.add_argument('--input',type=Path,default=root/'M19_Controlled_Blind_Input_30.csv')
    p.add_argument('--judge',type=Path,default=root/'QA_ChromaDB_LocalJudging_v2(2).py')
    p.add_argument('--model',required=True,help='Local multilingual-e5-large directory or model identifier')
    p.add_argument('--output',type=Path,default=root/'M19_Controlled_Reference_Scores.csv')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--expert-labels',type=Path,default=None,help='Optional CSV: case_id,expert_label_correct_1_error_0. Only expert-validated labels.')
    p.add_argument('--threshold',type=float,default=None,help='PRE-SPECIFIED score threshold: below => error; do not tune on same cases')
    args=p.parse_args()
    if not args.input.exists():p.error('Missing input: '+str(args.input))
    if not args.judge.exists():p.error('Missing judge: '+str(args.judge))
    if not 0<= (args.threshold if args.threshold is not None else 0.5)<=1:p.error('threshold must be 0..1')
    df=pd.read_csv(args.input,encoding='utf-8-sig',dtype={'case_id':str})
    req={'case_id','benchmark_id','question','reference_answer','answer'}
    if not req.issubset(df.columns):p.error('Input missing: '+str(req-set(df.columns)))
    if df.case_id.duplicated().any():p.error('Duplicate case_id')
    if df.empty:p.error('Empty input')
    judge=load_judge(args.judge)
    # Original judge expects 'answer' as REFERENCE and 'response' as GENERATED.
    source=df.rename(columns={'answer':'response','reference_answer':'answer'})
    scored=judge.evaluate(source,args.model,args.batch_size,args.no_download)
    scored=scored.rename(columns={'answer':'reference_answer','response':'answer'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    scored.to_csv(args.output,index=False,encoding='utf-8-sig')
    print('Saved:',args.output,'rows:',len(scored))
    print('Reference agreement mean:',round(float(pd.to_numeric(scored['v2_reference_agreement'],errors='coerce').mean()),4))
    print('IMPORTANT: these are reference-agreement proxies, not validated error detection metrics.')
    if args.expert_labels:
        if not args.expert_labels.exists():p.error('Missing expert-labels file')
        lab=pd.read_csv(args.expert_labels,encoding='utf-8-sig',dtype={'case_id':str})
        col='expert_label_correct_1_error_0'
        if not {'case_id',col}.issubset(lab.columns):p.error('Expert labels need case_id and '+col)
        if lab.case_id.duplicated().any():p.error('Duplicate expert case_id')
        m=scored.merge(lab[['case_id',col]],on='case_id',how='left',validate='one_to_one')
        y=pd.to_numeric(m[col],errors='coerce')
        if not y.notna().all() or not y.isin([0,1]).all():
            print('Expert labels incomplete or invalid. No diagnostic metrics reported.');return
        if args.threshold is None:
            print('No prespecified threshold: no classification metrics reported.');return
        pred=(m.v2_reference_agreement.astype(float)<args.threshold).astype(int)
        actual=(1-y).astype(int)
        tp=int(((pred==1)&(actual==1)).sum());fp=int(((pred==1)&(actual==0)).sum())
        tn=int(((pred==0)&(actual==0)).sum());fn=int(((pred==0)&(actual==1)).sum())
        precision=tp/(tp+fp) if tp+fp else None
        recall=tp/(tp+fn) if tp+fn else None
        f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
        print(json.dumps({'threshold':args.threshold,'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision_error':precision,'recall_error':recall,'f1_error':f1,'distinct_questions':m.benchmark_id.nunique(),'note':'Artificial perturbations; clustered by question; not independent samples.'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
