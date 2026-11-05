# -*- coding: utf-8 -*-
"""
M19_Advanced_Judges_Analyze.py
Compares the new judges with the existing ones on the V2F benchmark.

Inputs : merged scores/labels CSV (M19_V2F_Merged_Scores_Labels_v4.csv) + any of
         M19_V2F_NLI_Scores.csv, M19_V2F_LLM_Scores.csv, M19_V2F_Claim_Scores.csv
Outputs: M19_Advanced_AUC_table.csv, M19_Advanced_AUC_by_type.csv,
         M19_Advanced_paired_vs_reference.csv, M19_Advanced_ROC.png

Positive class = Correct. Cluster bootstrap by selection_id (2000 reps, seed 20261024).
Paired test: bootstrap difference AUC(judge) - AUC(reference judge) on the same resamples.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

SEED, REPS = 20261024, 2000
HERE = Path(__file__).resolve().parent


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if y.min() == y.max() or np.isnan(s).any():
        return np.nan
    return roc_auc_score(y, s)


def boot(df, cols, reps, mask=None):
    rng = np.random.default_rng(SEED)
    groups = {k: g.index.values for k, g in df.groupby('selection_id')}
    keys = list(groups)
    res = {c: [] for c in cols}
    for _ in range(reps):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        b = df.loc[idx]
        for c in cols:
            res[c].append(auc(b.y, b[c]))
    return {c: np.array(v) for c, v in res.items()}


def fmt(a, v):
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return f'{a:.3f} [{lo:.3f}, {hi:.3f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged', type=Path, default=HERE / 'M19_V2F_Merged_Scores_Labels_v4.csv')
    ap.add_argument('--nli', type=Path, default=HERE / 'M19_V2F_NLI_Scores.csv')
    ap.add_argument('--llm', type=Path, default=HERE / 'M19_V2F_LLM_Scores.csv')
    ap.add_argument('--claim', type=Path, default=HERE / 'M19_V2F_Claim_Scores.csv')
    ap.add_argument('--outdir', type=Path, default=HERE)
    ap.add_argument('--reps', type=int, default=REPS)
    a = ap.parse_args()

    m = pd.read_csv(a.merged, encoding='utf-8-sig')
    m['y'] = (m.human_label == 'Correct').astype(int)
    cols = {'v2_reference_agreement': 'Reference judge (E5)', 'evidence_judge_raw': 'Evidence judge (E5)',
            'v2_lexical_f1': 'Lexical F1', 'wc': 'Word count'}
    for path, mapping in [
        (a.nli, {'nli_score': 'NLI composite', 'ref_entail': 'NLI ref->ans entail',
                 'ev_entail_max': 'NLI evidence entail (max)'}),
        (a.llm, {'llm_ref_score': 'LLM judge (reference)', 'llm_ev_score': 'LLM judge (evidence)'}),
        (a.claim, {'claim_score': 'Claim-based judge', 'claim_precision': 'Claim precision',
                   'claim_recall': 'Claim recall'})]:
        if path.is_file():
            d = pd.read_csv(path, encoding='utf-8-sig')
            keep = ['case_id'] + [c for c in mapping if c in d.columns]
            m = m.merge(d[keep], on='case_id', how='left')
            cols.update({c: n for c, n in mapping.items() if c in d.columns})
    # NLI contradiction scores are "bad" signals; invert so that higher = more likely correct
    if 'ref_contra' in m.columns:
        pass
    cols = {c: n for c, n in cols.items() if c in m.columns and m[c].notna().all() and m[c].astype(str).ne('').all()}
    for c in cols:
        m[c] = m[c].astype(float)
    print('Judges analysed:', list(cols.values()))

    bs = boot(m, list(cols), a.reps)
    rows = []
    for c, n in cols.items():
        rows.append({'judge': n, 'column': c, 'AUC': round(auc(m.y, m[c]), 4),
                     'CI95_low': round(np.nanpercentile(bs[c], 2.5), 4),
                     'CI95_high': round(np.nanpercentile(bs[c], 97.5), 4), 'text': fmt(auc(m.y, m[c]), bs[c])})
    tab = pd.DataFrame(rows)
    tab.to_csv(a.outdir / 'M19_Advanced_AUC_table.csv', index=False, encoding='utf-8-sig')
    print(tab[['judge', 'text']].to_string(index=False))

    # paired difference vs reference judge and vs evidence judge
    ref = 'v2_reference_agreement'
    pr = []
    for c, n in cols.items():
        if c == ref:
            continue
        d = bs[c] - bs[ref]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        pr.append({'judge': n, 'delta_AUC_vs_reference': round(auc(m.y, m[c]) - auc(m.y, m[ref]), 4),
                   'CI95_low': round(np.nanpercentile(d, 2.5), 4), 'CI95_high': round(np.nanpercentile(d, 97.5), 4),
                   'p_two_sided_bootstrap': round(max(p, 1 / a.reps), 4)})
    pd.DataFrame(pr).to_csv(a.outdir / 'M19_Advanced_paired_vs_reference.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(pr).to_string(index=False))

    # per error type (Correct vs that single error type) and without Omission/Irrelevant
    types = ['Irrelevant', 'Numerical error', 'Conditional error', 'Contradiction', 'Omission']
    out = []
    for t in types + ['All errors', 'No Omission/Irrelevant']:
        if t == 'All errors':
            sub = m
        elif t.startswith('No '):
            sub = m[~m.response_type.isin(['Omission', 'Irrelevant'])]
        else:
            sub = m[m.response_type.isin(['Correct', t])]
        rec = {'subset': t, 'n_correct': int(sub.y.sum()), 'n_incorrect': int((1 - sub.y).sum())}
        for c, n in cols.items():
            rec[n] = round(auc(sub.y, sub[c]), 3)
        out.append(rec)
    pd.DataFrame(out).to_csv(a.outdir / 'M19_Advanced_AUC_by_type.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(out).to_string(index=False))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for c, n in cols.items():
        if c in ('wc', 'claim_precision', 'claim_recall', 'ref_entail', 'ev_entail_max'):
            continue
        fpr, tpr, _ = roc_curve(m.y, m[c])
        ax.plot(fpr, tpr, lw=1.6, label=f'{n} (AUC={auc(m.y, m[c]):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title('V2F: ROC of all judges (positive = Correct)')
    ax.legend(fontsize=7.5, loc='lower right'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(a.outdir / 'M19_Advanced_ROC.png', dpi=200)
    print('Saved outputs in', a.outdir)


if __name__ == '__main__':
    main()
