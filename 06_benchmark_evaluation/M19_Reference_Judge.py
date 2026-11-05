#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline reference-based judging for saved M19 Benchmark Runner CSV results.

Uses the ORIGINAL QA_ChromaDB_LocalJudging_v2.py evaluate() function unchanged.
No Groq or ChromaDB calls. No generated answers are modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {'answer', 'reference_answer', 'benchmark_id', 'version', 'mode', 'judge_raw'}
SCORES = ['v2_semantic_similarity', 'v2_lexical_f1', 'v2_reference_token_coverage',
          'v2_reference_agreement']


def load_original_judge(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Original judge not found: {path}')
    spec = importlib.util.spec_from_file_location('m19_local_judge_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Input not found: {path}')
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    elif path.suffix.lower() in {'.xlsx', '.xlsm'}:
        df = pd.read_excel(path)
    else:
        raise ValueError('Input must be CSV or XLSX.')
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    if df.empty:
        raise ValueError('Input has no rows.')
    return df


def pearson_safe(df: pd.DataFrame) -> float:
    valid = df[['judge_raw', 'v2_reference_agreement']].apply(pd.to_numeric, errors='coerce').dropna()
    if len(valid) < 3 or valid.nunique().min() < 2:
        return float('nan')
    return float(valid.corr().iloc[0, 1])


def safe_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors='coerce').dropna()
    return round(float(x.mean()), 5) if len(x) else float('nan')


def analyze(scored: pd.DataFrame, disagreement: float):
    scored = scored.copy()
    scored['judge_raw'] = pd.to_numeric(scored['judge_raw'], errors='coerce')
    scored['judge_difference'] = scored['judge_raw'] - scored['v2_reference_agreement']
    scored['absolute_judge_difference'] = scored['judge_difference'].abs()
    scored['judge_disagreement_flag'] = (
        scored['v2_status'].eq('success')
        & scored['absolute_judge_difference'].ge(disagreement)
    )
    scored['reference_unverified_flag'] = ~scored.get(
        'reference_verified', pd.Series('Unknown', index=scored.index)
    ).fillna('Unknown').astype(str).str.strip().str.lower().isin({'verified', 'yes', 'true', 'تاییدشده', 'تأییدشده'})
    scored['needs_human_review'] = (
        scored['v2_manual_review'].fillna(False).astype(bool)
        | scored['judge_disagreement_flag']
        | scored['reference_unverified_flag']
        | scored['v2_status'].ne('success')
    )

    valid = scored[scored['v2_status'].eq('success')].copy()
    records = []
    for (version, mode), part in valid.groupby(['version', 'mode'], dropna=False, sort=True):
        records.append({
            'version': version, 'mode': mode, 'runs': len(part),
            'distinct_questions': part['benchmark_id'].nunique(),
            'mean_evidence_judge_raw': safe_mean(part['judge_raw']),
            'mean_reference_agreement': safe_mean(part['v2_reference_agreement']),
            'mean_semantic_similarity': safe_mean(part['v2_semantic_similarity']),
            'mean_lexical_f1': safe_mean(part['v2_lexical_f1']),
            'mean_reference_token_coverage': safe_mean(part['v2_reference_token_coverage']),
            'mean_absolute_judge_difference': safe_mean(part['absolute_judge_difference']),
            'judge_correlation': pearson_safe(part),
            'judge_disagreement_count': int(part['judge_disagreement_flag'].sum()),
            'numeric_review_flags': int(part['v2_numeric_flag'].fillna('').ne('').sum()),
        })
    by_mode = pd.DataFrame(records)
    by_version = pd.DataFrame([{
        'version': version, 'runs': len(part), 'distinct_questions': part['benchmark_id'].nunique(),
        'mean_evidence_judge_raw': safe_mean(part['judge_raw']),
        'mean_reference_agreement': safe_mean(part['v2_reference_agreement']),
        'mean_absolute_judge_difference': safe_mean(part['absolute_judge_difference']),
        'judge_correlation': pearson_safe(part),
        'judge_disagreement_count': int(part['judge_disagreement_flag'].sum()),
        'numeric_review_flags': int(part['v2_numeric_flag'].fillna('').ne('').sum()),
    } for version, part in valid.groupby('version', sort=True)])

    review_cols = [c for c in [
        'benchmark_id', 'source_dataset', 'source_id', 'version', 'mode', 'question',
        'reference_answer', 'answer', 'reference_verified', 'prior_tuning_exposure',
        'judge_raw', *SCORES, 'v2_numeric_flag', 'v2_status',
        'absolute_judge_difference', 'judge_disagreement_flag', 'v2_manual_review',
        'reference_unverified_flag', 'needs_human_review'
    ] if c in scored.columns]
    review = scored.loc[scored['needs_human_review'], review_cols].copy()
    review = review.sort_values('absolute_judge_difference', ascending=False, na_position='last')
    review['human_reference_valid'] = ''
    review['human_answer_correctness_0_to_1'] = ''
    review['human_answer_completeness_0_to_1'] = ''
    review['human_critical_error'] = ''
    review['human_reviewer_notes'] = ''
    return scored, by_version, by_mode, review


def main():
    parser = argparse.ArgumentParser(description='Independent M19 reference judge for saved benchmark answers')
    parser.add_argument('--input', type=Path, default=Path('benchmark_results.csv'))
    parser.add_argument('--judge-script', type=Path, default=None,
                        help='Path to QA_ChromaDB_LocalJudging_v2.py; auto-detected next to this script')
    parser.add_argument('--model', default=None, help='Local multilingual-e5-large folder')
    parser.add_argument('--output', type=Path, default=None, help='Output XLSX file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--disagreement-threshold', type=float, default=0.20)
    parser.add_argument('--no-download', action='store_true', default=True,
                        help='Disallow model downloads (default: enabled)')
    parser.add_argument('--allow-download', action='store_true', help='Allow model name from HuggingFace')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check data and calculate lexical/numeric diagnostics without embeddings or final scoring')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')
    if not 0 <= args.disagreement_threshold <= 1:
        parser.error('--disagreement-threshold must be between 0 and 1')
    here = Path(__file__).resolve().parent
    original = args.judge_script
    if original is None:
        candidates = [here / 'QA_ChromaDB_LocalJudging_v2.py', here / 'QA_ChromaDB_LocalJudging_v2(2).py']
        original = next((p for p in candidates if p.is_file()), candidates[0])
    judge = load_original_judge(original)
    df = read_input(args.input)
    print(f'Loaded {len(df)} runs; {df.benchmark_id.nunique()} distinct questions; '
          f'versions={sorted(df.version.dropna().astype(str).unique())}')
    if 'status' in df.columns:
        print('Generation statuses:', df['status'].value_counts(dropna=False).to_dict())
    successful = df['status'].astype(str).eq('success') if 'status' in df.columns else pd.Series(True, index=df.index)
    working = df.loc[successful].copy()
    if working.empty:
        raise ValueError('No successful generated answers to evaluate.')
    if working['reference_answer'].isna().all():
        raise ValueError('All reference answers are missing.')
    if args.dry_run:
        invalid = [judge.is_invalid(a) for a in working['answer'].fillna('')]
        missing_refs = working['reference_answer'].isna() | working['reference_answer'].astype(str).str.strip().eq('')
        numeric = [bool(judge.number_tokens(str(r)) ^ judge.number_tokens(str(a)))
                   for r, a in zip(working['reference_answer'].fillna(''), working['answer'].fillna(''))]
        print('DRY RUN: embeddings not loaded; no reference-agreement scores generated.')
        print(f'Successful runs={len(working)}, missing references={int(missing_refs.sum())}, '
              f'invalid responses={sum(bool(x) for x in invalid)}, numeric review candidates={sum(numeric)}')
        return
    model = args.model or str(here / 'models' / 'multilingual-e5-large')
    if not args.allow_download and not Path(model).is_dir():
        raise FileNotFoundError(f'Embedding model folder not found: {model}\n'
                                'Use --model "C:/.../models/multilingual-e5-large"')
    print('Computing original LocalJudging_v2 reference scores using:', model)
    prepared = working.rename(columns={'answer': 'response', 'reference_answer': 'answer'})
    scored = judge.evaluate(prepared, model, args.batch_size, not args.allow_download)
    # Restore Benchmark Runner column names, without losing any existing data.
    scored = scored.rename(columns={'answer': 'reference_answer', 'response': 'answer'})
    if not scored.index.equals(working.index):
        raise RuntimeError('Unexpected index change while judging; refusing unsafe merge.')
    scored, by_version, by_mode, review = analyze(scored, args.disagreement_threshold)
    overview = pd.DataFrame([
        ['input_file', str(args.input)], ['original_judge_script', str(original)],
        ['embedding_model', model], ['all_input_runs', len(df)],
        ['evaluated_successful_runs', len(scored)],
        ['distinct_questions', scored['benchmark_id'].nunique()],
        ['reference_judge_success', int(scored['v2_status'].eq('success').sum())],
        ['reference_agreement_mean', safe_mean(scored.loc[scored['v2_status'].eq('success'), 'v2_reference_agreement'])],
        ['evidence_judge_mean', safe_mean(scored.loc[scored['v2_status'].eq('success'), 'judge_raw'])],
        ['overall_pearson_correlation', pearson_safe(scored.loc[scored['v2_status'].eq('success')])],
        ['judge_disagreement_threshold', args.disagreement_threshold],
        ['judge_disagreement_count', int(scored['judge_disagreement_flag'].sum())],
        ['human_review_rows', len(review)],
        ['unverified_reference_rows', int(scored['reference_unverified_flag'].sum())],
        ['method', 'Original LocalJudging_v2: 0.85 * E5 cosine + 0.15 * lexical F1'],
        ['scientific_limit', 'Reference agreement is not factual correctness; validate references and human labels.'],
        ['sample_limit', 'Repeated runs per question are not statistically independent.'],
    ], columns=['metric', 'value'])
    output = args.output or args.input.with_name(args.input.stem + '_Reference_Judged.xlsx')
    if output.resolve() == args.input.resolve():
        raise ValueError('Output cannot overwrite input.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        overview.to_excel(writer, sheet_name='Overview', index=False)
        by_version.to_excel(writer, sheet_name='By_Version', index=False)
        by_mode.to_excel(writer, sheet_name='By_Version_Mode', index=False)
        scored.to_excel(writer, sheet_name='All_Scored_Runs', index=False)
        review.to_excel(writer, sheet_name='Human_Review', index=False)
        # Make result sheets navigable and human-readable.
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        for ws in writer.book.worksheets:
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            ws.row_dimensions[1].height = 26
            for cell in ws[1]:
                cell.font = Font(bold=True, color='FFFFFF')
                cell.fill = PatternFill('solid', fgColor='183153')
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            for idx, cell in enumerate(ws[1], start=1):
                label = str(cell.value or '')
                width = 48 if label in {'question', 'answer', 'reference_answer', 'human_reviewer_notes'} else min(34, max(14, len(label) + 3))
                ws.column_dimensions[get_column_letter(idx)].width = width
    print('Saved:', output)
    print(overview.to_string(index=False))
    print('\nNote: This is a PROXY reference-agreement score, not validated factual accuracy.')


if __name__ == '__main__':
    main()
