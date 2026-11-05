# -*- coding: utf-8 -*-
"""
M19_Advanced_Judges.py
Three additional judges for the M19 V2F benchmark (resumable, append-only CSV output):

  nli    multilingual NLI / cross-encoder (local HuggingFace model)
         scores: ref_entail (reference -> answer), ref_entail_rev (answer -> reference),
                 ref_contra, ev_entail_max / ev_contra_max (fixed evidence chunks -> answer)
  llm    LLM-as-judge (Groq, openai/gpt-oss-120b), JSON verdict, two variants:
         llm_ref  (question + reference + answer)
         llm_ev   (question + fixed evidence chunks + answer; reference NOT shown)
  claim  claim-based judge (Groq): stage 1 extracts atomic claims from the answer and key
         points from the reference; stage 2 verifies each claim against reference+evidence and
         each reference point against the answer. Scores: precision, recall, contradiction rate.

Usage (from the '# New Approach' folder):
  python M19_Advanced_Judges.py nli   --input M19_V2F_Judge_Input.csv --evidence M19_Final_Fixed_Evidence_29_v2.csv --nli-model "models\\mdeberta-xnli" --output M19_V2F_NLI_Scores.csv
  python M19_Advanced_Judges.py llm   --input M19_V2F_Judge_Input.csv --evidence M19_Final_Fixed_Evidence_29_v2.csv --keys-file Keys.env --output M19_V2F_LLM_Scores.csv
  python M19_Advanced_Judges.py claim --input M19_V2F_Judge_Input.csv --evidence M19_Final_Fixed_Evidence_29_v2.csv --keys-file Keys.env --output M19_V2F_Claim_Scores.csv

Add --limit 5 for a trial run, --restart to delete the old output, --dry-run to check files only.
The judges never see the human labels. Evidence = the same fixed 7 chunks used by the evidence judge.
"""
import argparse, csv, hashlib, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLM_MODEL = os.getenv('M19_LLM_MODEL', 'openai/gpt-oss-120b')
PROMPT_VERSION = 'adv-judges-v1'


# ----------------------------------------------------------------------------- utilities
def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, encoding='utf-8-sig')


def load_inputs(inp, evidence, limit):
    df = read_csv(inp)
    need = {'case_id', 'benchmark_id', 'question', 'reference_answer', 'answer'}
    if not need.issubset(df.columns):
        raise SystemExit(f'Input is missing columns: {sorted(need - set(df.columns))}')
    ev = read_csv(evidence)
    chunks = {}
    for _, r in ev.iterrows():
        if str(r.get('status', 'success')) != 'success' and not str(r.get('retrieval_json', '')).strip():
            continue
        try:
            chunks[r['benchmark_id']] = json.loads(r['retrieval_json'])
        except Exception:
            chunks[r['benchmark_id']] = []
    missing = sorted(set(df.benchmark_id) - set(chunks))
    if missing:
        raise SystemExit(f'No fixed evidence for: {missing}')
    if limit:
        df = df.head(limit)
    return df, chunks


def done_ids(path):
    if not path.exists():
        return set()
    try:
        return set(read_csv(path)['case_id'])
    except Exception:
        return set()


def append_row(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Windows locks a CSV that is open in Excel: wait and retry instead of crashing
    for attempt in range(1, 61):
        try:
            new = not path.exists() or path.stat().st_size == 0
            with path.open('a', encoding='utf-8-sig' if new else 'utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                if new:
                    w.writeheader()
                w.writerow({k: row.get(k, '') for k in fields})
                f.flush(); os.fsync(f.fileno())
            return
        except PermissionError:
            print(f'  Cannot write {path.name} (open in Excel?). Close it; retry {attempt}/60 in 10s...', flush=True)
            time.sleep(10)
    raise SystemExit(f'Cannot write {path}. Close the file and rerun the same command (it resumes).')


def evidence_text(chunks, max_chars=900):
    parts = []
    for i, c in enumerate(chunks, 1):
        t = re.sub(r'\s+', ' ', str(c.get('text', ''))).strip()
        parts.append(f'[شاهد {i}] {t[:max_chars]}')
    return '\n'.join(parts)


# ----------------------------------------------------------------------------- NLI judge
NLI_FIELDS = ['case_id', 'benchmark_id', 'ref_entail', 'ref_neutral', 'ref_contra',
              'ref_entail_rev', 'ref_contra_rev', 'ev_entail_max', 'ev_contra_max',
              'nli_score', 'nli_model', 'prompt_version']


def run_nli(a):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    src = a.nli_model
    kw = {'local_files_only': True} if a.no_download else {}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    mdl = AutoModelForSequenceClassification.from_pretrained(src, **kw).eval()
    id2 = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    idx = {}
    for k, v in id2.items():
        if v.startswith('entail'): idx['e'] = k
        elif v.startswith('neutral'): idx['n'] = k
        elif v.startswith('contra'): idx['c'] = k
    if set(idx) != {'e', 'n', 'c'}:
        raise SystemExit(f'Cannot map NLI labels from config: {id2}')

    @torch.no_grad()
    def probs(pairs):
        out = []
        for i in range(0, len(pairs), a.batch_size):
            b = pairs[i:i + a.batch_size]
            enc = tok([p for p, _ in b], [h for _, h in b], truncation=True, max_length=512,
                      padding=True, return_tensors='pt')
            p = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()
            out.extend(p)
        return np.asarray(out)

    df, chunks = load_inputs(a.input, a.evidence, a.limit)
    out = Path(a.output)
    if a.restart and out.exists():
        out.unlink()
    done = done_ids(out)
    for _, r in df.iterrows():
        if r.case_id in done:
            continue
        ans = str(r.answer)
        ref = str(r.reference_answer)
        # question context is prepended so that short references remain interpretable
        q = str(r.question)
        prem_ref = f'{q} {ref}'
        hyp_ans = f'{q} {ans}'
        fw = probs([(prem_ref, hyp_ans)])[0]
        bw = probs([(hyp_ans, prem_ref)])[0]
        cs = chunks[r.benchmark_id]
        evp = probs([(str(c.get('text', ''))[:1500], ans) for c in cs]) if cs else np.zeros((1, 3))
        ev_e = float(evp[:, idx['e']].max())
        ev_c = float(evp[:, idx['c']].max())
        ref_e, ref_c = float(fw[idx['e']]), float(fw[idx['c']])
        rev_e, rev_c = float(bw[idx['e']]), float(bw[idx['c']])
        # pre-specified composite: bidirectional entailment minus contradiction (no tuning on labels)
        score = 0.5 * (ref_e + rev_e) - 0.5 * (ref_c + rev_c)
        append_row(out, NLI_FIELDS, {
            'case_id': r.case_id, 'benchmark_id': r.benchmark_id,
            'ref_entail': round(ref_e, 5), 'ref_neutral': round(float(fw[idx['n']]), 5),
            'ref_contra': round(ref_c, 5), 'ref_entail_rev': round(rev_e, 5),
            'ref_contra_rev': round(rev_c, 5), 'ev_entail_max': round(ev_e, 5),
            'ev_contra_max': round(ev_c, 5), 'nli_score': round(score, 5),
            'nli_model': src, 'prompt_version': PROMPT_VERSION})
        print(f'{r.case_id} ref_e={ref_e:.3f} ref_c={ref_c:.3f} ev_e={ev_e:.3f}', flush=True)


# ----------------------------------------------------------------------------- Groq helpers
def make_client(keys_file):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(keys_file, override=True)
    
    key = (os.getenv('GROQ_API_KEY', '').strip())
            

    if not key:
        raise SystemExit('Groq key missing in Keys.env')
    return OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1')


def parse_json(text):
    text = (text or '').strip()
    text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.S)
        if not m:
            raise ValueError('no JSON object in model output')
        return json.loads(m.group(0))


def _retry_wait(err, k):
    """Wait time: honour Groq 'try again in 1h2m3.4s' / Retry-After, else exponential backoff."""
    msg = str(err)
    m = re.search(r'try again in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?', msg)
    if m and any(m.groups()):
        h, mi, sec = (float(x) if x else 0.0 for x in m.groups())
        return min(7200, h * 3600 + mi * 60 + sec + 5)
    return min(120, 5 * (2 ** k))


def call_json(client, system, user, max_tokens=2500, retries=12):
    last = None
    for k in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL, temperature=0, max_tokens=max_tokens, timeout=120,
                messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                response_format={'type': 'json_object'})
            return parse_json(resp.choices[0].message.content), resp.usage.total_tokens if resp.usage else 0
        except Exception as e:  # rate limit, network loss, malformed JSON
            last = e
            wait = _retry_wait(e, k)
            print(f'  retry {k + 1}/{retries} in {wait:.0f}s: {type(e).__name__}: {str(e)[:160]}', flush=True)
            time.sleep(wait)
    raise SystemExit(f'LLM call failed after {retries} tries ({last}). Rerun the same command later; it resumes.')


# ----------------------------------------------------------------------------- LLM-as-judge
LLM_FIELDS = ['case_id', 'benchmark_id', 'llm_ref_verdict', 'llm_ref_score', 'llm_ref_error_type',
              'llm_ev_verdict', 'llm_ev_score', 'llm_ev_error_type', 'tokens', 'llm_model',
              'prompt_version', 'prompt_sha256']

LLM_SYSTEM = (
    'You are a strict evaluator of answers about Iran National Building Regulations (Mabhas 19). '
    'Judge ONLY factual and logical correctness relative to the given material, not style or length. '
    'An answer that is on topic but omits a required element, drops a condition or exception, changes '
    'a number/unit/threshold/comparison, or reverses a requirement is NOT correct. '
    'Output one JSON object only.')

LLM_REF_TMPL = """پرسش:
{q}

پاسخ مرجع (درست فرض می‌شود):
{ref}

پاسخ مورد ارزیابی:
{ans}

پاسخ مورد ارزیابی را فقط با توجه به پاسخ مرجع بسنج. خروجی JSON با این کلیدها:
{{"verdict": "correct" یا "incorrect",
  "score": عدد صحیح ۰ تا ۱۰۰ (احتمال این‌که پاسخ از نظر محتوا و منطق معادل مرجع و درست باشد),
  "error_type": یکی از "none","contradiction","conditional","numerical","omission","irrelevant",
  "reason": "یک جملهٔ کوتاه"}}"""

LLM_EV_TMPL = """پرسش:
{q}

شواهد بازیابی‌شده از متن مقررات:
{ev}

پاسخ مورد ارزیابی:
{ans}

پاسخ را فقط با توجه به شواهد بالا بسنج (پاسخ مرجع در اختیار تو نیست). خروجی JSON با این کلیدها:
{{"verdict": "correct" یا "incorrect",
  "score": عدد صحیح ۰ تا ۱۰۰ (احتمال این‌که پاسخ با شواهد سازگار، کامل و بدون خطای منطقی باشد),
  "error_type": یکی از "none","contradiction","conditional","numerical","omission","irrelevant",
  "reason": "یک جملهٔ کوتاه"}}"""


def _score(d):
    try:
        s = float(d.get('score'))
    except Exception:
        s = float('nan')
    return max(0.0, min(100.0, s)) / 100.0 if s == s else ''


def run_llm(a):
    client = make_client(a.keys_file)
    df, chunks = load_inputs(a.input, a.evidence, a.limit)
    out = Path(a.output)
    if a.restart and out.exists():
        out.unlink()
    done = done_ids(out)
    psha = sha256_text(LLM_SYSTEM + LLM_REF_TMPL + LLM_EV_TMPL)
    for _, r in df.iterrows():
        if r.case_id in done:
            continue
        ev = evidence_text(chunks[r.benchmark_id])
        d1, t1 = call_json(client, LLM_SYSTEM, LLM_REF_TMPL.format(q=r.question, ref=r.reference_answer, ans=r.answer))
        time.sleep(a.delay)
        d2, t2 = call_json(client, LLM_SYSTEM, LLM_EV_TMPL.format(q=r.question, ev=ev, ans=r.answer))
        append_row(out, LLM_FIELDS, {
            'case_id': r.case_id, 'benchmark_id': r.benchmark_id,
            'llm_ref_verdict': d1.get('verdict', ''), 'llm_ref_score': _score(d1),
            'llm_ref_error_type': d1.get('error_type', ''),
            'llm_ev_verdict': d2.get('verdict', ''), 'llm_ev_score': _score(d2),
            'llm_ev_error_type': d2.get('error_type', ''),
            'tokens': t1 + t2, 'llm_model': LLM_MODEL, 'prompt_version': PROMPT_VERSION, 'prompt_sha256': psha})
        print(f"{r.case_id} ref={d1.get('verdict')}/{_score(d1)} ev={d2.get('verdict')}/{_score(d2)}", flush=True)
        time.sleep(a.delay)


# ----------------------------------------------------------------------------- claim-based judge
CLAIM_FIELDS = ['case_id', 'benchmark_id', 'n_answer_claims', 'n_supported', 'n_contradicted',
                'n_unsupported', 'n_ref_points', 'n_ref_covered', 'n_ref_partial', 'n_ref_missing',
                'n_ref_contradicted', 'claim_precision', 'claim_recall', 'claim_contra_rate',
                'claim_score', 'claims_json', 'tokens', 'llm_model', 'prompt_version', 'prompt_sha256']

CLAIM_SYSTEM = (
    'You are a careful legal-text analyst for Iran National Building Regulations (Mabhas 19). '
    'Work at the level of atomic claims. Conditions, exceptions, quantities, units, thresholds and '
    'comparison directions are part of a claim: a claim that drops or changes them is not supported. '
    'Output one JSON object only.')

CLAIM_EXTRACT_TMPL = """پرسش:
{q}

پاسخ مرجع:
{ref}

پاسخ مورد ارزیابی:
{ans}

کار ۱: پاسخ مورد ارزیابی را به گزاره‌های اتمی و مستقل بشکن (حداکثر ۸ گزاره؛ شرط‌ها، استثناها، عددها و واحدها را در همان گزاره نگه دار).
کار ۲: پاسخ مرجع را به نکات کلیدی ضروری بشکن (حداکثر ۶ نکته).
خروجی JSON: {{"answer_claims": ["..."], "reference_points": ["..."]}}"""

CLAIM_VERIFY_TMPL = """پرسش:
{q}

پاسخ مرجع:
{ref}

شواهد بازیابی‌شده:
{ev}

پاسخ مورد ارزیابی (کامل):
{ans}

گزاره‌های استخراج‌شده از پاسخ:
{claims}

نکات کلیدی مرجع:
{points}

کار ۱: برای هر گزاره‌ی پاسخ، وضعیت را مشخص کن: "supported" (با مرجع یا شواهد سازگار و کامل است)، "contradicted" (با مرجع یا شواهد در تضاد است یا شرط/عدد/جهت مقایسه را تغییر می‌دهد)، "unsupported" (نه تأیید می‌شود نه رد؛ بی‌ربط یا بدون پشتوانه).
کار ۲: برای هر نکته‌ی مرجع، در پاسخ کامل بررسی کن: "covered" (به‌طور کامل با شرط‌ها و عددها)، "partial", "missing", یا "contradicted".
خروجی JSON: {{"claims": [{{"i": 1, "status": "...", "note": "..."}}], "points": [{{"i": 1, "status": "...", "note": "..."}}]}}"""


def run_claim(a):
    client = make_client(a.keys_file)
    df, chunks = load_inputs(a.input, a.evidence, a.limit)
    out = Path(a.output)
    if a.restart and out.exists():
        out.unlink()
    done = done_ids(out)
    psha = sha256_text(CLAIM_SYSTEM + CLAIM_EXTRACT_TMPL + CLAIM_VERIFY_TMPL)
    for _, r in df.iterrows():
        if r.case_id in done:
            continue
        ev = evidence_text(chunks[r.benchmark_id])
        d1, t1 = call_json(client, CLAIM_SYSTEM, CLAIM_EXTRACT_TMPL.format(q=r.question, ref=r.reference_answer, ans=r.answer))
        time.sleep(a.delay)
        claims = [str(c) for c in d1.get('answer_claims', [])][:8]
        points = [str(p) for p in d1.get('reference_points', [])][:6]
        if not claims:
            claims = [str(r.answer)]
        if not points:
            points = [str(r.reference_answer)]
        d2, t2 = call_json(client, CLAIM_SYSTEM, CLAIM_VERIFY_TMPL.format(
            q=r.question, ref=r.reference_answer, ev=ev, ans=r.answer,
            claims='\n'.join(f'{i}. {c}' for i, c in enumerate(claims, 1)),
            points='\n'.join(f'{i}. {p}' for i, p in enumerate(points, 1))))
        cs = [str(x.get('status', '')).lower() for x in d2.get('claims', [])]
        ps = [str(x.get('status', '')).lower() for x in d2.get('points', [])]
        # pad missing verdicts conservatively
        cs += ['unsupported'] * (len(claims) - len(cs)); cs = cs[:len(claims)]
        ps += ['missing'] * (len(points) - len(ps)); ps = ps[:len(points)]
        ns, nc, nu = cs.count('supported'), cs.count('contradicted'), cs.count('unsupported')
        rc, rp, rm, rx = ps.count('covered'), ps.count('partial'), ps.count('missing'), ps.count('contradicted')
        precision = ns / len(cs)
        recall = (rc + 0.5 * rp) / len(ps)
        contra = (nc + rx) / (len(cs) + len(ps))
        f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        # pre-specified composite: F1 of claim precision/recall, reduced by contradiction rate
        score = max(0.0, f - contra)
        append_row(out, CLAIM_FIELDS, {
            'case_id': r.case_id, 'benchmark_id': r.benchmark_id,
            'n_answer_claims': len(cs), 'n_supported': ns, 'n_contradicted': nc, 'n_unsupported': nu,
            'n_ref_points': len(ps), 'n_ref_covered': rc, 'n_ref_partial': rp, 'n_ref_missing': rm,
            'n_ref_contradicted': rx, 'claim_precision': round(precision, 5), 'claim_recall': round(recall, 5),
            'claim_contra_rate': round(contra, 5), 'claim_score': round(score, 5),
            'claims_json': json.dumps({'claims': claims, 'points': points, 'claim_status': cs, 'point_status': ps},
                                      ensure_ascii=False),
            'tokens': t1 + t2, 'llm_model': LLM_MODEL, 'prompt_version': PROMPT_VERSION, 'prompt_sha256': psha})
        print(f'{r.case_id} P={precision:.2f} R={recall:.2f} contra={contra:.2f} score={score:.2f}', flush=True)
        time.sleep(a.delay)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('judge', choices=['nli', 'llm', 'claim'])
    p.add_argument('--input', type=Path, default=HERE / 'M19_V2F_Judge_Input.csv')
    p.add_argument('--evidence', type=Path, default=HERE / 'M19_Final_Fixed_Evidence_29_v2.csv')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keys-file', type=Path, default=HERE / 'Keys.env')
    p.add_argument('--nli-model', default='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
                   help='HF id or local folder of a 3-class NLI model (entailment/neutral/contradiction)')
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--delay', type=float, default=1.0, help='seconds between Groq calls')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--restart', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    for f in (a.input, a.evidence):
        if not f.is_file():
            raise SystemExit(f'Missing file: {f}')
    if a.dry_run:
        df, chunks = load_inputs(a.input, a.evidence, a.limit)
        print(f'OK: {len(df)} responses, {len(chunks)} evidence sets, judge={a.judge}')
        return
    {'nli': run_nli, 'llm': run_llm, 'claim': run_claim}[a.judge](a)
    print('Finished ->', a.output)


if __name__ == '__main__':
    main()
