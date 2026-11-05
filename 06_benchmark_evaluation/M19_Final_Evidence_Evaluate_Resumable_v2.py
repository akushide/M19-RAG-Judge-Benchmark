#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-evidence controlled audit; resumable with per-case checkpoints and progress."""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, re, time, os
from pathlib import Path
import numpy as np

FUNCTIONS={'normalize','token_set','lexical_f1','invalid_answer','numeric_values',
           '_strip_list_prefix','strip_synthetic_citations','_audit_clean',
           '_remove_reference_identifiers','substantive_numbers','semantic_judge'}

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

def load_exact_judge(path):
    source=path.read_text(encoding='utf-8-sig');tree=ast.parse(source)
    defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in FUNCTIONS]
    found={n.name for n in defs}
    if found!=FUNCTIONS:raise RuntimeError('Missing judge dependencies: '+str(FUNCTIONS-found))
    module=ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[]))
    env={'np':np,'re':re}
    constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_CIT_NUM','_CIT_LABEL','_CIT_ITEM','_CIT_GROUP','_CIT_INLINE'} for t in n.targets)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=constants,type_ignores=[])),str(path),'exec'),env)
    exec(compile(module,str(path),'exec'),env)
    return env['semantic_judge'],hashlib.sha256(source.encode('utf-8')).hexdigest()

def main():
    p=argparse.ArgumentParser(description='Controlled test: fixed-evidence judge (resumable)')
    root=Path(__file__).resolve().parent
    p.add_argument('--input',type=Path,default=root/'M19_Final_Benchmark_Judge_Input.csv')
    p.add_argument('--benchmark',type=Path,default=root/'benchmark_results.csv')
    p.add_argument('--script',type=Path,default=root/'Interactive_RAG_24_multi_witness_audit.py')
    p.add_argument('--version',default='24')
    p.add_argument('--mode',default='Deeper',choices=['Instant','Deep','Deeper'])
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--output',type=Path,default=root/'M19_Final_Evidence_Scores.csv')
    p.add_argument('--manifest',type=Path,default=root/'M19_Final_Evidence_Manifest.json')
    p.add_argument('--torch-threads',type=int,default=4,help='Limit CPU threads; default 4')
    p.add_argument('--restart',action='store_true',help='Ignore prior output and score all cases again')
    a=p.parse_args()
    for path in (a.input,a.benchmark,a.script):
        if not path.is_file():p.error('File not found: '+str(path))
    cases=read_csv(a.input);rows=read_csv(a.benchmark)
    if not cases or len({x['case_id'] for x in cases})!=len(cases):p.error('Expected non-empty input with unique case IDs')
    matches=[r for r in rows if str(r.get('version'))==a.version and r.get('mode')==a.mode and r.get('status')=='ok']
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

    if a.no_download and not a.model.is_dir():p.error('Local model directory not found: '+str(a.model))
    import torch
    torch.set_num_threads(max(1,a.torch_threads))
    from sentence_transformers import SentenceTransformer
    print('Loading local embedding model...',flush=True)
    model=SentenceTransformer(str(a.model),local_files_only=True) if a.no_download else SentenceTransformer(str(a.model))
    print('Loading exact semantic_judge...',flush=True)
    judge,sha=load_exact_judge(a.script)

    fieldnames=['case_id','benchmark_id','evidence_judge_raw','evidence_numeric_warning',
      'evidence_unverified_numbers','evidence_best_source_id','evidence_max_semantic',
      'evidence_lexical_f1','fixed_evidence_run_key','fixed_evidence_source_ids','evidence_judge_script_sha256']
    done={}
    if a.output.exists() and not a.restart:
        try:
            for r in read_csv(a.output):
                if r.get('case_id') and r.get('benchmark_id'):
                    done[r['case_id']]=r
            print(f'Resuming from existing output: {len(done)} case(s) already present.',flush=True)
        except Exception as e:
            print(f'Could not read previous output ({e}); starting fresh.',flush=True)
            done={}
    output_rows=[done[x['case_id']] for x in cases if x['case_id'] in done]
    started=time.time()
    total=len(cases)
    for i,case in enumerate(cases,1):
        if case['case_id'] in done:
            continue
        bid=case['benchmark_id'];ev=fixed[bid]
        print(f'[{i}/{total}] Scoring {case["case_id"]} ({bid}) ...',flush=True)
        t0=time.time()
        score,details=judge(case['answer'],ev['chunks'],model)
        row={'case_id':case['case_id'],'benchmark_id':bid,
          'evidence_judge_raw':score,
          'evidence_numeric_warning':int(bool(details.get('numeric_warning'))),
          'evidence_unverified_numbers':json.dumps(details.get('unverified_numbers',[]),ensure_ascii=False),
          'evidence_best_source_id':details.get('best_source_id',''),
          'evidence_max_semantic':details.get('max_semantic',''),
          'evidence_lexical_f1':details.get('lexical_f1',''),
          'fixed_evidence_run_key':ev['run_key'],
          'fixed_evidence_source_ids':json.dumps([x['id'] for x in ev['chunks']],ensure_ascii=False),
          'evidence_judge_script_sha256':sha}
        done[case['case_id']]=row
        output_rows=[done[x['case_id']] for x in cases if x['case_id'] in done]
        write_csv(a.output,output_rows,fieldnames)
        print(f'    saved checkpoint; case time {time.time()-t0:.1f}s; total elapsed {(time.time()-started)/60:.1f} min.',flush=True)

    manifest={'note':'Evidence fixed per QUESTION from historical benchmark. No new retrieval or generation.',
      'evidence_version':a.version,'evidence_mode':a.mode,'judge_script':a.script.name,
      'judge_script_sha256':sha,'model':str(a.model),'cases':len(output_rows),
      'questions':{k:{'run_key':v['run_key'],'source_ids':[c['id'] for c in v['chunks']],
      'evidence_sha256':hashlib.sha256(json.dumps(v['chunks'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
      for k,v in fixed.items()}}
    a.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',a.output,'rows:',len(output_rows),'questions:',len(fixed),flush=True)
    print('Saved manifest:',a.manifest,flush=True)
    print('Mean evidence agreement:',round(float(np.mean([float(x['evidence_judge_raw']) for x in output_rows])),5),flush=True)
    print('No labels used during scoring. Do NOT treat evidence agreement as factual truth.',flush=True)

if __name__=='__main__':
    main()
