#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-evidence controlled audit using EXACT semantic_judge from a selected RAG version.
No Groq calls, no fresh retrieval, no Streamlit launch, no label leakage.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, re
from pathlib import Path
import numpy as np

FUNCTIONS={'normalize','token_set','lexical_f1','invalid_answer','numeric_values',
           '_strip_list_prefix','strip_synthetic_citations','_audit_clean',
           '_remove_reference_identifiers','substantive_numbers','semantic_judge'}

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def load_exact_judge(path):
    source=path.read_text(encoding='utf-8-sig');tree=ast.parse(source)
    defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in FUNCTIONS]
    found={n.name for n in defs}
    if found!=FUNCTIONS:raise RuntimeError('Missing judge dependencies: '+str(FUNCTIONS-found))
    module=ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[]))
    env={'np':np,'re':re}
    # Citation regex constants are required by strip_synthetic_citations.
    constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_CIT_NUM','_CIT_LABEL','_CIT_ITEM','_CIT_GROUP','_CIT_INLINE'} for t in n.targets)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=constants,type_ignores=[])),str(path),'exec'),env)
    exec(compile(module,str(path),'exec'),env)
    return env['semantic_judge'],hashlib.sha256(source.encode('utf-8')).hexdigest()

def main():
    p=argparse.ArgumentParser(description='Controlled test: fixed-evidence judge')
    root=Path(__file__).resolve().parent
    p.add_argument('--input',type=Path,default=root/'M19_Controlled_Blind_Input_30.csv')
    p.add_argument('--benchmark',type=Path,default=root/'benchmark_results.csv')
    p.add_argument('--script',type=Path,default=root/'Interactive_RAG_24_multi_witness_audit.py')
    p.add_argument('--version',default='24',help='Which benchmark version supplies FIXED evidence')
    p.add_argument('--mode',default='Deeper',choices=['Instant','Deep','Deeper'])
    p.add_argument('--model',required=True,help='Same local embedding model as benchmark')
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--output',type=Path,default=root/'M19_Controlled_Evidence_Scores.csv')
    p.add_argument('--manifest',type=Path,default=root/'M19_Controlled_Evidence_Manifest.json')
    a=p.parse_args()
    for path in (a.input,a.benchmark,a.script):
        if not path.is_file():p.error('File not found: '+str(path))
    cases=read_csv(a.input);rows=read_csv(a.benchmark)
    if len(cases)!=30 or len({x['case_id'] for x in cases})!=30:p.error('Expected 30 unique case IDs')
    matches=[r for r in rows if str(r.get('version'))==a.version and r.get('mode')==a.mode and r.get('status')=='ok']
    # Historical runner may use success status; accept a nonempty retrieval record instead.
    if not matches:matches=[r for r in rows if str(r.get('version'))==a.version and r.get('mode')==a.mode and r.get('retrieval_json')]
    fixed={}
    for r in matches:
        bid=r['benchmark_id']
        if bid in fixed:raise RuntimeError('Ambiguous fixed evidence for '+bid)
        chunks=json.loads(r['retrieval_json'])
        if not isinstance(chunks,list) or not chunks or any('text' not in x or 'id' not in x for x in chunks):
            raise RuntimeError('Invalid evidence for '+bid)
        fixed[bid]={'chunks':chunks,'run_key':r['run_key']}
    missing={x['benchmark_id'] for x in cases}-set(fixed)
    if missing:p.error('Missing fixed evidence for: '+str(missing))
    from sentence_transformers import SentenceTransformer
    if a.no_download and not Path(a.model).is_dir():p.error('Local model directory not found: '+a.model)
    model=SentenceTransformer(a.model,local_files_only=True) if a.no_download else SentenceTransformer(a.model)
    judge,sha=load_exact_judge(a.script)
    out=[]
    for case in cases:
        bid=case['benchmark_id'];ev=fixed[bid]
        score,details=judge(case['answer'],ev['chunks'],model)
        out.append({'case_id':case['case_id'],'benchmark_id':bid,
                    'evidence_judge_raw':score,
                    'evidence_numeric_warning':int(bool(details.get('numeric_warning'))),
                    'evidence_unverified_numbers':json.dumps(details.get('unverified_numbers',[]),ensure_ascii=False),
                    'evidence_best_source_id':details.get('best_source_id',''),
                    'evidence_max_semantic':details.get('max_semantic',''),
                    'evidence_lexical_f1':details.get('lexical_f1',''),
                    'fixed_evidence_run_key':ev['run_key'],
                    'fixed_evidence_source_ids':json.dumps([x['id'] for x in ev['chunks']],ensure_ascii=False),
                    'evidence_judge_script_sha256':sha})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    manifest={'note':'Evidence fixed per QUESTION from historical benchmark. No new retrieval or generation.',
              'evidence_version':a.version,'evidence_mode':a.mode,'judge_script':a.script.name,
              'judge_script_sha256':sha,'model':a.model,'cases':len(out),
              'questions':{k:{'run_key':v['run_key'],'source_ids':[c['id'] for c in v['chunks']],
                              'evidence_sha256':hashlib.sha256(json.dumps(v['chunks'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
                           for k,v in fixed.items()}}
    a.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',a.output,'rows:',len(out),'questions:',len(fixed))
    print('Saved manifest:',a.manifest)
    print('Mean evidence agreement:',round(float(np.mean([x['evidence_judge_raw'] for x in out])),5))
    print('No labels used during scoring. Do NOT treat evidence agreement as factual truth.')
if __name__=='__main__':main()
