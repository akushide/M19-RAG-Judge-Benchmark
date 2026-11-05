# -*- coding: utf-8 -*-
"""Reproducible batch runner for original M19 Streamlit RAG v20/v23/v24.

It loads original function bodies via AST, without executing Streamlit UI,
SQLite feedback handlers, or module-level side effects. No changes are made to
retrieval, prompts, generation or auditing algorithms.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = {20: 'Interactive_RAG_20_condition_audit.py',
           23: 'Interactive_RAG_23_domain_audit.py',
           24: 'Interactive_RAG_24_multi_witness_audit.py'}
DEPTH_K = {'Instant': 3, 'Deep': 5, 'Deeper': 7}
DEPTH_MAX_TOKENS = {'Instant': 1000, 'Deep': 1800, 'Deeper': 2800}
FIELDS = ['run_key','timestamp_utc','benchmark_id','split','source_dataset','source_id',
          'question_type_auto','question','reference_answer','reference_verified',
          'prior_tuning_exposure','version','mode','K','status','error','answer',
          'latency_seconds','retrieval_seconds','generation_seconds','audit_seconds',
          'prompt_tokens','completion_tokens','total_tokens','generation_calls',
          'continuation_calls','answer_complete','finish_reason','source_ids_json',
          'retrieval_json','text_support_score','numeric_consistency_score',
          'numeric_claim_flags','claim_audit_json','semantic_consistency',
          'semantic_flags_json','judge_raw','judge_details_json',
          'script_sha256','benchmark_sha256','llm_model','embedding_model']


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read_xlsx(path, sheet_name='Benchmark_100'):
    """Read cell values with standard library only; supports normal Excel XLSX."""
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
          'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
          'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('m:si',ns):
                strings.append(''.join(t.text or '' for t in si.findall('.//m:t',ns)))
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        sheets=wb.findall('m:sheets/m:sheet',ns)
        sh=next((s for s in sheets if s.get('name')==sheet_name),None)
        if sh is None: raise ValueError(f'Sheet {sheet_name!r} not found: {[s.get("name") for s in sheets]}')
        rid=sh.get('{'+ns['r']+'}id')
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rel=next(x for x in rels if x.get('Id')==rid)
        target=rel.get('Target').lstrip('/')
        if not target.startswith('xl/'):
            target='xl/'+target
        root=ET.fromstring(z.read(target))
        rows=[]
        for row in root.findall('m:sheetData/m:row',ns):
            values={}
            for cell in row.findall('m:c',ns):
                letters=re.match(r'[A-Z]+',cell.get('r','')).group()
                idx=0
                for ch in letters: idx=idx*26+ord(ch)-64
                typ=cell.get('t')
                if typ=='inlineStr':
                    value=''.join(t.text or '' for t in cell.findall('.//m:t',ns))
                else:
                    v=cell.find('m:v',ns)
                    value='' if v is None else v.text or ''
                    if typ=='s' and value: value=strings[int(value)]
                values[idx-1]=value
            rows.append(values)
        if not rows: raise ValueError('Empty worksheet')
        header=rows[0]
        result=[]
        for row in rows[1:]:
            obj={name:row.get(i,'') for i,name in header.items() if name}
            if obj.get('question','').strip(): result.append(obj)
        return result


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_env(path):
    from dotenv import load_dotenv
    if path.is_file(): load_dotenv(path,override=False)  # optional: a GROQ_API_KEY environment variable also works
    key=os.getenv('GROQ_API_KEY','').strip()
    if not key: raise RuntimeError('Groq key missing in Keys.env')
    os.environ['GROQ_API_KEY']=key


def load_original_logic(script, model_name, embed_path, max_context):
    """Compile original functions/regex constants only, no Streamlit import/execution."""
    tree=ast.parse(script.read_text(encoding='utf-8'),filename=str(script))
    keep_functions={
        'normalize','token_set','numeric_values','lexical_f1','invalid_answer',
        'search','compose_context','_merge_continuation','_looks_unfinished',
        'answer_question','strip_synthetic_citations','clean_display_citations',
        '_remove_reference_identifiers','_strip_list_prefix','_audit_clean',
        'substantive_numbers','claim_segments','_evidence_sentences',
        'audit_claim_support','_normalized_semantic','_polarity','_relation',
        '_conditional_risks','_regulatory_markers','_topic_tokens','_same_subject',
        '_structured_proposition','_bound_relation_class','_domain_facets',
        '_best_evidence_candidates','_split_compound_claim','_evidence_alignment_reason',
        'semantic_consistency_audit','semantic_judge'
    }
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in keep_functions:
            node.decorator_list=[]
            nodes.append(node)
        elif isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) for t in node.targets):
            names=[t.id for t in node.targets]
            if names and all(n.startswith('_') for n in names):
                nodes.append(node)
    namespace={'__builtins__':__builtins__, 're':re,'json':json,'np':np,
               'logger':logging.getLogger('m19-benchmark'),
               'MAX_CONTEXT':max_context,'LLM_MODEL':model_name,
               'EMBED_MODEL':embed_path,'DEPTH_MAX_TOKENS':DEPTH_MAX_TOKENS}
    compiled=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(script),'exec')
    exec(compiled,namespace)
    return namespace


def load_resources(chroma_path,collection_name,embed_path):
    import chromadb
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    if not chroma_path.is_dir() or not (chroma_path/'chroma.sqlite3').is_file():
        raise FileNotFoundError(f'ChromaDB not found: {chroma_path}')
    if not embed_path.is_dir(): raise FileNotFoundError(f'Embedding model not found: {embed_path}')
    client=chromadb.PersistentClient(path=str(chroma_path))
    collection=client.get_collection(collection_name)
    embedder=SentenceTransformer(str(embed_path))
    llm=OpenAI(api_key=os.environ['GROQ_API_KEY'],base_url='https://api.groq.com/openai/v1')
    return collection,embedder,llm


def append_csv(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    new=not path.exists() or path.stat().st_size==0
    with path.open('a',encoding='utf-8-sig' if new else 'utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore')
        if new: writer.writeheader()
        writer.writerow({k:record.get(k,'') for k in FIELDS})
        f.flush();os.fsync(f.fileno())


def existing_keys(path):
    if not path.exists(): return set()
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return {row['run_key'] for row in csv.DictReader(f) if row.get('status') in ('success','not_found','truncated')}


def make_key(bench_id,version,mode,script_hash,bench_hash,model_name,context):
    raw=json.dumps([bench_id,version,mode,script_hash,bench_hash,model_name,context],ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def run_one(row,version,mode,logic,collection,embedder,model_name,script_hash,bench_hash,context):
    started=time.perf_counter()
    k=DEPTH_K[mode]
    rec={'timestamp_utc':utcnow(),'benchmark_id':row.get('benchmark_id',''),
         'split':row.get('split',''),'source_dataset':row.get('source_dataset',''),
         'source_id':row.get('source_id',''),'question_type_auto':row.get('question_type_auto',''),
         'question':row['question'],'reference_answer':row.get('reference_answer',''),
         'reference_verified':row.get('reference_verified',''),
         'prior_tuning_exposure':row.get('prior_tuning_exposure',''),
         'version':version,'mode':mode,'K':k,'status':'error','error':'',
         'script_sha256':script_hash,'benchmark_sha256':bench_hash,
         'llm_model':model_name,'embedding_model':str(context['embed_path'])}
    rec['run_key']=make_key(rec['benchmark_id'],version,mode,script_hash,bench_hash,model_name,str(context))
    try:
        t=time.perf_counter()
        sources=logic['search'](row['question'],collection,embedder,k)
        rec['retrieval_seconds']=round(time.perf_counter()-t,3)
        rec['source_ids_json']=json.dumps([x['id'] for x in sources],ensure_ascii=False)
        rec['retrieval_json']=json.dumps(sources,ensure_ascii=False,default=str)
        t=time.perf_counter()
        answer,usage=logic['answer_question'](row['question'],sources,mode)
        rec['generation_seconds']=round(time.perf_counter()-t,3)
        rec['answer']=answer
        for field in ('prompt_tokens','completion_tokens','total_tokens','generation_calls',
                      'continuation_calls','answer_complete','finish_reason'):
            rec[field]=usage.get(field)
        if logic['invalid_answer'](answer): rec['status']='not_found'
        elif not usage.get('answer_complete',False): rec['status']='truncated'
        else: rec['status']='success'
        t=time.perf_counter()
        support,nflags,audit=logic['audit_claim_support'](answer,sources)
        rec['text_support_score']=support
        rec['numeric_claim_flags']=nflags
        rec['numeric_consistency_score']=audit.get('numeric_consistency_score')
        rec['claim_audit_json']=json.dumps(audit,ensure_ascii=False,default=str)
        if version==20:
            sem,detail=logic['semantic_consistency_audit'](answer,sources)
        else:
            sem,detail=logic['semantic_consistency_audit'](answer,sources,audit)
        rec['semantic_consistency']=sem
        rec['semantic_flags_json']=json.dumps(detail,ensure_ascii=False,default=str)
        raw,judge_details=logic['semantic_judge'](answer,sources,embedder)
        rec['judge_raw']=raw
        rec['judge_details_json']=json.dumps(judge_details,ensure_ascii=False,default=str)
        rec['audit_seconds']=round(time.perf_counter()-t,3)
    except Exception as exc:
        rec['status']='error'
        rec['error']=f'{type(exc).__name__}: {exc}'
        logging.exception('Failed: %s v%s %s',row.get('benchmark_id'),version,mode)
    rec['latency_seconds']=round(time.perf_counter()-started,3)
    return rec


def main(argv=None):
    parser=argparse.ArgumentParser(description='M19 automatic RAG benchmark (v20, v23, v24)')
    parser.add_argument('--benchmark',type=Path,default=HERE/'M19_Benchmark_100_Candidate.xlsx')
    parser.add_argument('--scripts-dir',type=Path,default=HERE)
    parser.add_argument('--keys-file',type=Path,default=HERE/'Keys.env')
    parser.add_argument('--output',type=Path,default=HERE/'benchmark_results.csv')
    parser.add_argument('--versions',nargs='+',type=int,choices=[20,23,24],default=[20,23,24])
    parser.add_argument('--modes',nargs='+',choices=list(DEPTH_K),default=list(DEPTH_K))
    parser.add_argument('--limit',type=int,default=5,help='Number of benchmark questions (0 = all)')
    parser.add_argument('--split',default='all',choices=['all','Development','Held-out candidate'])
    parser.add_argument('--delay',type=float,default=2.0,help='Seconds between requests')
    parser.add_argument('--max-context',type=int,default=10500)
    parser.add_argument('--dry-run',action='store_true',help='Check files and plan; no Groq/Chroma requests')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.limit<0 or args.delay<0: parser.error('limit/delay must be nonnegative')
    for v in args.versions:
        if not (args.scripts_dir/SCRIPTS[v]).is_file(): parser.error(f'Missing original script: {SCRIPTS[v]}')
    if not args.benchmark.is_file(): parser.error(f'Missing benchmark: {args.benchmark}')
    rows=read_xlsx(args.benchmark)
    if args.split!='all': rows=[r for r in rows if r.get('split')==args.split]
    if args.limit: rows=rows[:args.limit]
    if not rows: parser.error('No questions selected')
    bench_hash=sha256(args.benchmark)
    script_hashes={v:sha256(args.scripts_dir/SCRIPTS[v]) for v in args.versions}
    base=Path(os.getenv('M19_BASE_PATH',str(args.scripts_dir)))
    chroma_path=Path(os.getenv('M19_CHROMA_PATH',str(base/'M19RDBEmbeding_ML_E5_L_ChromaDB_indexing')))
    collection_name=os.getenv('M19_COLLECTION','chunks_ML_E5_L_ChromaDB')
    embed_path=Path(os.getenv('M19_EMBED_MODEL',str(base/'models'/'multilingual-e5-large')))
    llm_name=os.getenv('M19_LLM_MODEL','openai/gpt-oss-120b')
    context={'embed_path':embed_path,'chroma_path':chroma_path,'collection':collection_name,
             'max_context':args.max_context,'llm_model':llm_name}
    plan=[(r,v,m) for r in rows for v in args.versions for m in args.modes]
    done=existing_keys(args.output)
    pending=[(r,v,m) for r,v,m in plan if make_key(r.get('benchmark_id',''),v,m,script_hashes[v],bench_hash,llm_name,str(context)) not in done]
    print(f'Questions: {len(rows)} | Planned: {len(plan)} | Remaining: {len(pending)}')
    print(f'Benchmark: {args.benchmark} | SHA256: {bench_hash[:16]}')
    print(f'Output: {args.output} | Chroma: {chroma_path} | Model: {embed_path}')
    for v in args.versions: print(f'  v{v} SHA256 {script_hashes[v][:16]}')
    if args.dry_run:
        print('DRY RUN: no model, database, or network calls.');return 0
    load_env(args.keys_file)
    collection,embedder,llm=load_resources(chroma_path,collection_name,embed_path)
    logics={}
    for v in args.versions:
        logic=load_original_logic(args.scripts_dir/SCRIPTS[v],llm_name,str(embed_path),args.max_context)
        logic['load_llm']=lambda client=llm: client
        logics[v]=logic
    total=len(pending)
    for idx,(row,v,mode) in enumerate(pending,1):
        rec=run_one(row,v,mode,logics[v],collection,embedder,llm_name,script_hashes[v],bench_hash,context)
        append_csv(args.output,rec)
        print(f'[{idx}/{total}] {row.get("benchmark_id")} v{v} {mode}: '
              f'{rec["status"]}, {rec["latency_seconds"]}s, '
              f'judge={rec.get("judge_raw", "-")}',flush=True)
        if idx<total and args.delay: time.sleep(args.delay)
    print('Completed. CSV:',args.output)
    return 0


if __name__=='__main__':
    sys.exit(main())
