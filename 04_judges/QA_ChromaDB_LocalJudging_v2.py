#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline RAG response evaluator for Persian M19 Q&A.

Dependencies: pip install pandas openpyxl numpy sentence-transformers torch tqdm
No Groq/API/ChromaDB calls. Scores are proxies for reference agreement, NOT
independent proof of regulatory correctness. Review numeric/technical claims.

Examples:
 python QA_ChromaDB_LocalJudging_v2.py --input M19DB1385_503.xlsx --model "C:/.../multilingual-e5-large"
 python QA_ChromaDB_LocalJudging_v2.py --input M19DB2127_498.xlsx --model "C:/.../multilingual-e5-large" --review M19_RAG_50_Content_Review_Completed2.xlsx
 python QA_ChromaDB_LocalJudging_v2.py --input M19DB1385_503.xlsx --model intfloat/multilingual-e5-large --no-download
"""
from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

INVALID_FULL = [
    r'^\s*(?:پاسخ\s*(?:مناسب\s*)?(?:یافت\s*نشد|پیدا\s*نشد)|اطلاعات\s*(?:کافی\s*)?(?:یافت\s*نشد|موجود\s*نیست)|no\s+answer|not\s+found|error\b|خطا(?:\s*[:：].*)?)\s*[.!؟]*\s*$',
    r'^\s*(?:متاسفانه|متأسفانه)?\s*(?:در\s*متن|در\s*منابع)?\s*(?:پاسخی|اطلاعاتی)\s*(?:یافت\s*نشد|وجود\s*ندارد)\s*[.!؟]*\s*$',
]
PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
STOP = set('از به در با بر که این آن یک و یا را برای است هستند شود شده می باشد بود نیز هر تا چه چگونه آیا کدام چرا اگر اما همچنین های ها تر ترین خود مورد صورت باید فقط همه تمام پس نسبت عنوان طبق'.split())
# Only defensible lexical equivalences. Do NOT equate energy with power, EUI, efficiency, etc.
ALIASES = [
    {'راندمان','بازده','بازدهی'}, {'پایش','مانیتورینگ'},
    {'عایق کاری','عایق بندی'}, {'گرمایش','سیستم گرمایشی','سامانه گرمایشی'},
    {'سرمایش','سیستم سرمایشی','سامانه سرمایشی'},
    {'پنجره','بازشو نورگذر'}, {'کربن نهفته','کربن تجسم یافته'},
    {'گاز گلخانه ای','گازهای گلخانه ای'},
]


def norm(s: object) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ''
    s = str(s).translate(PERSIAN_DIGITS).lower()
    s = s.replace('ي','ی').replace('ك','ک').replace('ة','ه').replace('ۀ','ه')
    s = re.sub('[\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\u00a0]', ' ', s)
    s = re.sub(r'[\u064b-\u065f\u0670]', '', s)
    s = re.sub(r'[^\w\s.%/+-]', ' ', s, flags=re.UNICODE)
    return ' '.join(s.split())


def words(s: str) -> set[str]:
    return {t for t in norm(s).split() if len(t) > 2 and t not in STOP}


def canon_alias(s: str) -> str:
    s = ' ' + norm(s) + ' '
    for group in ALIASES:
        variants = sorted((norm(x) for x in group), key=len, reverse=True)
        canonical = variants[0]
        for v in variants:
            s = re.sub(r'(?<!\w)' + re.escape(v) + r'(?!\w)', canonical, s)
    return s.strip()


def token_f1(reference: str, response: str) -> tuple[float,float,float]:
    a, b = words(canon_alias(reference)), words(canon_alias(response))
    if not a or not b:
        return 0., 0., 0.
    shared = len(a & b)
    recall, precision = shared / len(a), shared / len(b)
    return (2*recall*precision/(recall+precision) if shared else 0.), recall, precision


def number_tokens(s: str) -> set[str]:
    s = norm(s)
    # Numerals and percentages; normalize decimals, and exclude tiny standalone digit markers.
    found = re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)?\s*%?(?!\w)', s)
    out = set()
    for item in found:
        item = item.strip().replace(',', '.')
        try:
            pct = item.endswith('%')
            num = float(item.rstrip('%'))
            out.add(f'{num:g}' + ('%' if pct else ''))
        except ValueError:
            pass
    return out


def is_invalid(response: str) -> str | None:
    s = norm(response)
    if not s or s in {'nan','none','null','-'}:
        return 'empty'
    for pat in INVALID_FULL:
        if re.fullmatch(pat, s, re.IGNORECASE):
            return 'not_found'
    if re.match(r'^(?:error|exception|traceback|apierror|rate limit|خطا)(?:\s|$)', s):
        return 'error'
    return None


def embeddings(texts: list[str], model_path: str, batch_size: int, no_download: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    path = Path(model_path)
    if no_download and not path.exists():
        raise FileNotFoundError('Local model path does not exist; --no-download forbids network downloads.')
    model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    # multilingual-e5 models expect passage:/query: prefixes; use symmetric passage for answer comparison.
    inputs = ['passage: ' + (t if t else ' ') for t in texts]
    return np.asarray(model.encode(inputs, batch_size=batch_size, normalize_embeddings=True,
                                   show_progress_bar=True), dtype=np.float32)


def evaluate(df: pd.DataFrame, model_path: str, batch_size: int, no_download: bool) -> pd.DataFrame:
    col = {str(c).strip().lower():c for c in df.columns}
    if not {'answer','response'} <= set(col):
        raise ValueError('Required columns: answer and response')
    refs = [norm(x) for x in df[col['answer']].tolist()]
    resps = [norm(x) for x in df[col['response']].tolist()]
    statuses = [is_invalid(x) or ('missing_reference' if not r else 'success') for r,x in zip(refs,resps)]
    valid_ids = [i for i,s in enumerate(statuses) if s == 'success']
    unique_texts = list(dict.fromkeys([refs[i] for i in valid_ids] + [resps[i] for i in valid_ids]))
    vectors = embeddings(unique_texts, model_path, batch_size, no_download) if unique_texts else np.empty((0,0))
    lookup = dict(zip(unique_texts, vectors))
    sim_scores=[]; lexical_scores=[]; coverage_scores=[]; num_flags=[]; scores=[]; needs_review=[]
    for r,x,status in zip(refs,resps,statuses):
        if status != 'success':
            sim_scores.append(0.); lexical_scores.append(0.); coverage_scores.append(0.)
            num_flags.append(''); scores.append(0.); needs_review.append(status == 'missing_reference')
            continue
        sim = float(np.clip(np.dot(lookup[r],lookup[x]),0,1))
        f1, coverage, _ = token_f1(r,x)
        ref_nums, resp_nums = number_tokens(r), number_tokens(x)
        missing = ref_nums - resp_nums
        extra = resp_nums - ref_nums
        flag = 'different_numbers' if missing and extra else 'reference_numbers_missing' if missing else 'additional_numbers_check' if extra else ''
        # Calibrated-looking scores are deliberately avoided: continuous proxy, no supervised fitting.
        # Similarity is dominant; lexical overlap is a weak supporting signal, never a zero-score veto.
        score = 0.85*sim + 0.15*f1
        # Numeric discrepancies are REVIEW flags, not unconditional contradictions: answer may add detail.
        score = float(np.clip(score,0,1))
        sim_scores.append(round(sim,5)); lexical_scores.append(round(f1,5))
        coverage_scores.append(round(coverage,5)); num_flags.append(flag)
        scores.append(round(score,5)); needs_review.append(bool(flag) or score < .65)
    out = df.copy()
    for name, values in {
        'v2_status':statuses, 'v2_semantic_similarity':sim_scores, 'v2_lexical_f1':lexical_scores,
        'v2_reference_token_coverage':coverage_scores, 'v2_numeric_flag':num_flags,
        'v2_reference_agreement':scores, 'v2_manual_review':needs_review,
    }.items():
        out[name] = values
    return out


def review_report(scored: pd.DataFrame, review_file: Path) -> pd.DataFrame:
    review = pd.read_excel(review_file, sheet_name='50_Content_Reviews')
    # Prefer dataset+ID, never fuzzy-join based on question text alone.
    required = {'Dataset','ID','Reviewer_Correctness','Reviewer_Completeness'}
    if not required.issubset(review.columns):
        raise ValueError('Review file missing: ' + ', '.join(sorted(required-set(review.columns))))
    if 'id' not in {str(c).strip().lower() for c in scored.columns}:
        raise ValueError('Input needs id column for matching reviewed rows.')
    id_col = next(c for c in scored.columns if str(c).strip().lower() == 'id')
    review = review.copy()
    review['ID'] = pd.to_numeric(review['ID'],errors='coerce')
    for c in ['Reviewer_Correctness','Reviewer_Completeness']:
        review[c] = pd.to_numeric(review[c],errors='coerce')
    # The review set covers two datasets; caller's input file must match exactly one dataset.
    data = scored.copy(); data['ID'] = pd.to_numeric(data[id_col],errors='coerce')
    # Disambiguate IDs using input name at call-site: filter passed dataframe already.
    selected = review[review['Dataset'].astype(str).str.casefold() == str(scored.attrs.get('dataset','')).casefold()]
    if selected.empty:
        return pd.DataFrame({'message':['No review rows for this dataset.']})
    if selected['ID'].duplicated().any() or data['ID'].duplicated().any():
        raise ValueError('Duplicate IDs found; cannot reliably match review records.')
    matched = selected.merge(data[['ID','v2_reference_agreement','v2_semantic_similarity','v2_numeric_flag']],on='ID',how='left',validate='one_to_one')
    matched['difference_vs_correctness'] = matched['v2_reference_agreement'] - matched['Reviewer_Correctness']
    matched['absolute_error'] = matched['difference_vs_correctness'].abs()
    return matched


def main():
    p = argparse.ArgumentParser(description='Offline M19 reference-agreement evaluator v2')
    p.add_argument('--input',required=True,type=Path,help='Existing Excel with answer and response columns')
    p.add_argument('--output',type=Path,help='Default: <input stem>_Judged_v2.xlsx')
    p.add_argument('--model',default='multilingual-e5-large',help='Local SentenceTransformer model path (preferred) or HF name')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--no-download',action='store_true',help='Require --model to be an existing local path')
    p.add_argument('--review',type=Path,help='Optional completed 50-row reviewer workbook')
    args = p.parse_args()
    if args.batch_size < 1: p.error('--batch-size must be positive')
    if not args.input.exists(): p.error(f'Input file not found: {args.input}')
    output = args.output or args.input.with_name(args.input.stem+'_Judged_v2.xlsx')
    if output.resolve() == args.input.resolve(): p.error('Output must not overwrite input')
    df = pd.read_excel(args.input)
    result = evaluate(df,args.model,args.batch_size,args.no_download)
    result.attrs['dataset'] = re.match(r'M19DB\d+',args.input.stem,re.IGNORECASE).group(0) if re.match(r'M19DB\d+',args.input.stem,re.IGNORECASE) else args.input.stem
    counts = result['v2_status'].value_counts()
    valid = result.loc[result['v2_status']=='success','v2_reference_agreement']
    summary = pd.DataFrame([
        ['input',str(args.input)],['embedding_model',args.model],['total_rows',len(result)],
        ['success',int(counts.get('success',0))],['empty',int(counts.get('empty',0))],
        ['not_found',int(counts.get('not_found',0))],['error',int(counts.get('error',0))],
        ['missing_reference',int(counts.get('missing_reference',0))],
        ['mean_reference_agreement_success',round(float(valid.mean()),5) if len(valid) else None],
        ['manual_review_count',int(result['v2_manual_review'].sum())],
        ['scoring_method','0.85 * E5 cosine (clipped 0..1) + 0.15 * lexical F1'],
        ['interpretation','Proxy for reference agreement; not independent factual/regulatory correctness'],
        ['timestamp',datetime.now().isoformat(timespec='seconds')]
    ],columns=['metric','value'])
    with pd.ExcelWriter(output,engine='openpyxl') as writer:
        result.to_excel(writer,sheet_name='Judged_v2',index=False)
        summary.to_excel(writer,sheet_name='Summary',index=False)
        if args.review:
            if not args.review.exists(): p.error(f'Review file not found: {args.review}')
            matched = review_report(result,args.review)
            matched.to_excel(writer,sheet_name='Reviewer_Comparison',index=False)
            if 'absolute_error' in matched:
                numeric = matched['absolute_error'].dropna()
                stats = pd.DataFrame([['matched_review_scores',len(numeric)],['MAE_vs_reviewer_correctness',float(numeric.mean()) if len(numeric) else None]],columns=['metric','value'])
                stats.to_excel(writer,sheet_name='Reviewer_Stats',index=False)
    print('Saved:',output)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
