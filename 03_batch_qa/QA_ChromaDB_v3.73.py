# RAG + Evaluation Pipeline for M19 (Groq API)
# - Reads questions from Excel
# - Answers via Groq API (OpenAI-compatible)
# - Evaluates with semantic similarity + LLM-as-Judge
# - Saves results with timing and scores
# - Includes progress bar (tqdm)
# - Saves settings and averages in Excel
#  Handles following errors and rate limits properly
# ایرادت مشاهده شده در خروجی برنامه
#شباهت معنایی کاذب برای پاسخ‌های "پاسخ یافت نشد"	قبل از محاسبه‌ی شباهت، پاسخ را بررسی کنید و اگر حاوی "یافت نشد" یا "ERROR" بود، sim=0.0 قرار دهید.	بالا
#قضاوت نادرست (Judge)	دستورالعمل قضاوت را کوتاه‌تر و دقیق‌تر کنید. در صورت خطا، از sim به‌عنوان جایگزین استفاده کنید.	بالا
#پاسخ‌های مرجع (Ground Truth) کوتاه و کلی	پاسخ‌های مرجع را به ۲-۳ جمله کامل و مستند بازنویسی کنید.	متوسط
#سوالات کلی و گنگ	سوالات را خاص‌تر کنید و به بافت مبحث نوزدهم ارجاع دهید.	متوسط
#محدودیت توکن و نرخ در Groq	retry_limit را افزایش دهید، max_tokens را کاهش دهید و از مدل‌های سریع‌تر استفاده کنید.	متوسط
#ذخیره‌سازی تنظیمات و میانگین‌ها	در کد نهایی، تنظیمات و میانگین‌ها را در فایل خروجی ذخیره کنید (همان‌طور که قبلاً انجام دادید).	پایین
# =====================================================
# This is a reminder that the Llama 3.3 70B Versatile model will be decommissioned on August 16, 2026. After the decommission date, requests to the model will no longer be served.
# We encourage you to transition your workloads to our recommended replacement model, GPT OSS 120B / Qwen3.6 27B, as soon as possible.
# No LLM judge; works
# make better Judge
#top_k=3 --> 10 and Judge > 0
# Improve Answering!
# Improve Judging!


import os
import time
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import torch
from tqdm import tqdm
from openai import OpenAI

# =========================
#  تنظیمات اولیه (ثابت)
# =========================

PY_FILE = os.path.basename(__file__)

# --- Groq API key: read ONLY from the environment (never hard-code it) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError("Set the GROQ_API_KEY environment variable (see Keys.env.example).")

# --- مدل‌ها ---
MODEL_EMBEDDING_NAME = "multilingual-e5-large"

MODEL_ANSWER = "openai/gpt-oss-120b"
MODEL_ANSWER2= "gpt-oss-120b"
               #  "llama-3.1-8b-instant"
               # "openai/gpt-oss-120b"
               # "qwen/qwen3.6-27b"
               # "openai/gpt-oss-20b"
               # "openai/gpt-oss-safeguard-20b"

INPUT_FILE = "M19DB2127.xlsx"
#INPUT_FILE = "M19DB1385.xlsx"

#INPUT_FILE = "M19DB2127_100.xlsx"
#INPUT_FILE = "M19DB1385_100.xlsx"

#INPUT_FILE = "M19DB2127_20.xlsx"
#INPUT_FILE = "M19DB1385_20.xlsx"


TEMPERATURE = 0.3
MAX_TOKENS = 512
TOP_P = 0.9

FPATH = os.environ.get("M19_BASE_PATH", str(Path(__file__).resolve().parent))
FMODEL = "ML_E5_L"
CHROMA_DB_PATH = os.path.join(FPATH, f"M19RDBEmbeding_{FMODEL}_ChromaDB_indexing")
COLLECTION_NAME = f"chunks_{FMODEL}_ChromaDB"

EMBEDDING_MODEL_PATH = os.path.join(FPATH, "models", MODEL_EMBEDDING_NAME)
OUTPUT_FILE = f"{INPUT_FILE}_Answers_Groq_{FMODEL}_Embeding_ChromaDB_indexing___LLManswering_{MODEL_ANSWER2}.xlsx"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

print("=" * 50)
print("🔄 Loading Embedding Model...")
embed_model = SentenceTransformer(
    EMBEDDING_MODEL_PATH,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("✅ Embedding Model loaded.")

print("🔄 Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
collection = chroma_client.get_collection(COLLECTION_NAME)
print(f"🔢 Number of documents in ChromaDB: {collection.count()}")

print("🔄 Groq client ready.")
print("=" * 50)

# =========================
#  جستجو در ChromaDB
# =========================

def search(query, top_k=5):
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    retrieved = []
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        retrieved.append({
            "id": results['ids'][0][i],
            "text": doc,
            "metadata": meta,
            "score": 1 - dist
        })
    return retrieved

# =========================
#  پرسش از Groq
# =========================

def ask_question_groq(query, top_k=5):
    retrieved = search(query, top_k)
    if not retrieved:
        return "پاسخ در اطلاعات موجود یافت نشد.", []

    context = "\n\n---\n\n".join([f"[بخش {i+1}]\n{r['text']}" for i, r in enumerate(retrieved)])

    MAX_CONTEXT_LEN = 2500
    if len(context) > MAX_CONTEXT_LEN:
        context = context[:MAX_CONTEXT_LEN] + "..."

    system_prompt = (
        "شما یک کارشناس خبره در استخراج داده‌های فنی از «مبحث نوزدهم مقررات ملی ساختمان» هستید. "
        "تمام پاسخ‌های خود را به زبان فارسی روان و رسمی بنویسید. "
        "پاسخ را تا حد امکان بر اساس اطلاعات زیر و با استناد به متن ارائه دهید. "
        "اگر در متن، هیچ سرنخ، مفهوم یا عبارت مرتبط با سؤال وجود ندارد، فقط در آن صورت بگویید «پاسخ در اطلاعات موجود یافت نشد.» "
        "اگر حتی بخشی از پاسخ در متن وجود دارد، با استفاده از همان بخش، پاسخ کوتاه، دقیق و فنی در ۱ تا ۳ جمله بنویسید. "
        "از تکرار جملات خودداری کنید و از مقدمه‌هایی مانند «بر اساس متن» یا «با توجه به» استفاده نکنید؛ مستقیماً با اصل پاسخ شروع کنید."
    )

    user_prompt = f"""### اطلاعات:

{context}

### سوال:
{query}

### پاسخ (به فارسی):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_ANSWER,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            timeout=60.0,
        )
        answer_text = completion.choices[0].message.content.strip()
        if "### پاسخ (به فارسی):" in answer_text:
            answer_text = answer_text.split("### پاسخ (به فارسی):")[-1].strip()
        return answer_text, retrieved
    except Exception as e:
        tqdm.write(f"❌ Error in ask_question_groq: {type(e).__name__}: {e}")
        return f"❌ خطا در ارتباط با Groq: {e}", []

# =========================
#  شباهت معنایی
# =========================

def semantic_similarity_fa(text1, text2):
    if not text1 or not text2:
        return 0.0
    try:
        v1 = embed_model.encode(text1, normalize_embeddings=True)
        v2 = embed_model.encode(text2, normalize_embeddings=True)

        sim = cosine_similarity([v1], [v2])[0][0]

        len_ratio = min(len(text2) / max(len(text1), 1), 1.0)
        sim = sim * (0.6 + 0.4 * len_ratio)

        if len(text2.split()) < 5:
            sim *= 0.7

        return float(sim)
    except Exception as e:
        tqdm.write(f"❌ Error in semantic_similarity_fa: {e}")
        tqdm.write(f"   text1[:100]: {text1[:100] if text1 else ''}")
        tqdm.write(f"   text2[:100]: {text2[:100] if text2 else ''}")
        return 0.0

# =========================
#  مفاهیم دامنه‌ای
# =========================

DOMAIN_SYNONYMS = {
    "انرژی": ["توان", "مصرف", "شدت مصرف", "eui", "بازده", "کارایی"],
    "مصرف": ["هدررفت", "اتلاف", "مصرف انرژی", "بهینه‌سازی"],
    "حرارتی": ["گرما", "دما", "آسایش حرارتی", "thermal", "انتقال حرارت"],
    "روشنایی": ["نور", "illuminance", "daylight", "artificial", "light"],
    "جریان": ["دبی", "flow", "ultrasonic", "مافوق صوت"],
    "کنترل": ["مدیریت", "تنظیم", "bms", "lms", "ibms", "ems"],
    "پوسته": ["envelope", "glazing", "opaque", "پوسته خارجی", "پوسته نورگذر"],
    "بار": ["load", "ظرفیت", "partial", "بار جزئی"],
    "دوقلوی دیجیتال": ["digital", "twin", "مدل مجازی", "مدل پویا"],
}

IMPLICIT_SYNONYMS = {
    "پایش": ["نظارت", "مانیتورینگ"],
    "بهینه‌سازی": ["کاهش مصرف", "افزایش بهره‌وری"],
    "کارایی": ["بازده", "راندمان"],
}

DOMAIN_CONTRADICTIONS_STRONG = [
    ("افزایش مصرف", "کاهش مصرف"),
    ("هدررفت انرژی", "صرفه‌جویی انرژی"),
    ("گرم", "سرد"),
    ("روشن", "خاموش"),
    ("مسکونی", "غیرمسکونی"),
]

DOMAIN_CONTRADICTIONS_WEAK = [
    ("زیاد", "کم"),
    ("بالا", "پایین"),
]

STOPWORDS_FA = {"است", "می‌شود", "می‌گردد", "برای", "به", "از", "که", "این", "آن", "در", "با", "و"}

def normalize_token(tok: str) -> str:
    tok = tok.strip().lower()
    for suf in ["ها", "های", "ات", "ی"]:
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            tok = tok[:-len(suf)]
    return tok

def expand_with_domain_synonyms(tokens):
    expanded = set(tokens)
    for t in list(tokens):
        for key, syns in DOMAIN_SYNONYMS.items():
            if t == key or t in syns:
                expanded.add(key)
                expanded.update(syns)

        if t in IMPLICIT_SYNONYMS:
            expanded.update(IMPLICIT_SYNONYMS[t])

    return expanded

def extract_concepts(text: str):
    tokens = []
    for w in text.split():
        w = normalize_token(w)
        if len(w) > 2 and w not in STOPWORDS_FA:
            tokens.append(w)
    expanded = expand_with_domain_synonyms(tokens)
    return expanded

# =========================
#  Judge مفهومی جدید
# =========================

def semantic_concept_judge_pro_plus(ground_truth: str, model_answer: str) -> float:
    if not ground_truth or not model_answer:
        return 0.0

    gt = ground_truth.strip().lower()
    ans = model_answer.strip().lower()

    invalid_phrases = [
        "پاسخ یافت نشد",
        "اطلاعات موجود یافت نشد",
        "پاسخ در اطلاعات موجود یافت نشد",
        "یافت نشد",
        "error",
        "خطا",
    ]
    if any(p in ans for p in invalid_phrases):
        return 0.0

    gt_concepts = extract_concepts(gt)
    ans_concepts = extract_concepts(ans)

    for a, b in DOMAIN_CONTRADICTIONS_STRONG:
        if a in ans and b in gt:
            return 0.0
        if b in ans and a in gt:
            return 0.0

    weak_penalty = 0.0
    for a, b in DOMAIN_CONTRADICTIONS_WEAK:
        if a in ans and b in gt:
            weak_penalty += 0.05
        if b in ans and a in gt:
            weak_penalty += 0.05

    total_weight = 0
    covered_weight = 0

    for c in gt_concepts:
        w = 1
        if c in DOMAIN_SYNONYMS:
            w = 3
        total_weight += w
        if c in ans_concepts:
            covered_weight += w

    coverage_ratio = covered_weight / total_weight if total_weight else 0.0

    if len(ans.split()) > len(gt.split()) * 1.3:
        coverage_ratio += 0.1

    if coverage_ratio >= 0.75:
        base_score = 1.0
    elif coverage_ratio >= 0.50:
        base_score = 0.75
    elif coverage_ratio >= 0.30:
        base_score = 0.50
    else:
        base_score = 0.0

    score = max(0.0, base_score - weak_penalty)
    return score

def final_judge(ground_truth: str, model_answer: str) -> float:
    sim = semantic_similarity_fa(ground_truth, model_answer)
    concept_score = semantic_concept_judge_pro_plus(ground_truth, model_answer)
    final_score = 0.65 * concept_score + 0.35 * sim
    return float(final_score)

# =========================
#  AnalysisAll به‌روزرسانی
# =========================

def update_analysis_file(
    py_file,
    output_file,
    input_file,
    model_embedding,
    model_answer,
    avg_retrieval_time,
    avg_response_time,
    avg_semantic_sim,
    avg_judge_score,
    avg_final_score,
    successful_count,
    total_questions,
    not_found_count,
    error_count
):
    analysis_file = "AnalysisAll.xlsx"

    new_row = {
        "row": None,
        "Py File": py_file,
        "OUTPUT_FILE": output_file,
        "INPUT_FILE": input_file,
        "MODEL_EMBEDDING_NAME": model_embedding,
        "MODEL_ANSWER": model_answer,
        "retrieval time Average (s)": avg_retrieval_time,
        "response time Average (s)": avg_response_time,
        "semantic similarity Average": avg_semantic_sim,
        "judge score Average": avg_judge_score,
        "final score Average": avg_final_score,
        "Successful count": successful_count,
        "Total questions": total_questions,
        "Not found count'": not_found_count,
        "Error count": error_count
    }

    if os.path.exists(analysis_file):
        try:
            df_existing = pd.read_excel(analysis_file, sheet_name="نتایج عملی")
            new_row["row"] = len(df_existing) + 1
            df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"⚠️ خطا در خواندن فایل AnalysisAll: {e}")
            new_row["row"] = 1
            df_new = pd.DataFrame([new_row])
    else:
        new_row["row"] = 1
        df_new = pd.DataFrame([new_row])

    with pd.ExcelWriter(analysis_file, engine='openpyxl') as writer:
        df_new.to_excel(writer, sheet_name="نتایج عملی", index=False)

    print(f"✅ ردیف جدید به '{analysis_file}' اضافه شد.")

# =========================
#  تابع اصلی اجرای Batch
# =========================

def run_batch_questions(
    input_file: str,
    output_file: str,
    top_k: int = 5,
    retry_limit: int = 7,
    save_interval: int = 5
):
    if not os.path.exists(input_file):
        tqdm.write(f"❌ فایل ورودی '{input_file}' پیدا نشد!")
        return

    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    df.columns = [c.strip().lower() for c in df.columns]
    required_cols = ["question", "answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"فایل ورودی باید شامل ستون '{col}' باشد.")

    if os.path.exists(output_file):
        tqdm.write("📂 Found existing output file. Resuming...")
        if output_file.endswith(".xlsx"):
            df_out = pd.read_excel(output_file)
        else:
            df_out = pd.read_csv(output_file)
        df_out.columns = [c.strip().lower() for c in df_out.columns]
    else:
        tqdm.write("🆕 Starting fresh benchmark...")
        df_out = df.copy()
        extra_cols = [
            "response",
            "retrieval_time_sec",
            "response_time_sec",
            "semantic_similarity",
            "judge_score",
            "final_score",
            "status"
        ]
        for col in extra_cols:
            if col not in df_out.columns:
                df_out[col] = ""

    total = len(df_out)
    tqdm.write(f"\n🚀 Starting benchmark for {total} questions...")

    for idx, row in tqdm(df_out.iterrows(), total=total, desc="Processing Questions", unit="q"):
        q_raw = row.get("question", "")
        if isinstance(q_raw, float) and np.isnan(q_raw):
            continue
        question = str(q_raw).strip()
        if not question or question.lower() == "nan":
            continue

        ground_truth = str(row.get("answer", "")).strip()
        existing_resp = str(row.get("response", "")).strip()
        existing_judge = row.get("judge_score", "")

        judge_exists = (
            pd.notna(existing_judge)
            and str(existing_judge).strip() != ""
            and str(existing_judge).strip().lower() != "nan"
        )

        status_prev = str(row.get("status", "")).strip().lower()

        # فقط خطاها و not_found را دوباره بررسی کن؛ success را نگه دار
        if status_prev == "success" and judge_exists and float(existing_judge) > 0:
            tqdm.write(f"⏩ Skipping question {idx+1} (previous success with judge>0)")
            continue

        answer = None
        for attempt in range(1, retry_limit + 1):
            try:
                t1 = time.time()
                retrieved = search(question, top_k=top_k)
                t2 = time.time()
                if not retrieved:
                    raise Exception("No relevant documents found.")

                t3 = time.time()
                answer, _ = ask_question_groq(question, top_k=top_k)
                t4 = time.time()

                is_error = answer.startswith("❌") or "خطا" in answer
                is_not_found = "پاسخ در اطلاعات موجود یافت نشد" in answer

                if is_not_found:
                    sim = 0.0
                    judge = 0.0
                    final_score = 0.0
                    status = "not_found"
                    tqdm.write(f"ℹ️ Q{idx+1}: Answer not found, marked as 'not_found'.")
                elif is_error:
                    sim = 0.0
                    judge = 0.0
                    final_score = 0.0
                    status = "error"
                    tqdm.write(f"⚠️ Q{idx+1}: API Error, scores set to 0.")
                else:
                    if ground_truth:
                        sim = semantic_similarity_fa(ground_truth, answer)
                        judge = semantic_concept_judge_pro_plus(ground_truth, answer)
                        final_score = final_judge(ground_truth, answer)
                    else:
                        sim = 0.0
                        judge = 0.0
                        final_score = 0.0
                    status = "success"

                df_out.at[idx, "response"] = answer
                df_out.at[idx, "retrieval_time_sec"] = round(t2 - t1, 3)
                df_out.at[idx, "response_time_sec"] = round(t4 - t3, 3)
                df_out.at[idx, "semantic_similarity"] = float(sim)
                df_out.at[idx, "judge_score"] = float(judge)
                df_out.at[idx, "final_score"] = float(final_score)
                df_out.at[idx, "status"] = status

                tqdm.write(f"✅ Q{idx+1}: Sim={sim:.3f}, Judge={judge:.3f}, Final={final_score:.3f}")
                break

            except Exception as e:
                err_msg = str(e).lower()
                if "rate limit" in err_msg or "429" in err_msg or "413" in err_msg:
                    wait_time = 60.0
                    match = re.search(r"try again in ([0-9\.]+)s", err_msg)
                    if match:
                        wait_time = max(float(match.group(1)), 60.0)
                    tqdm.write(f"⚠️ Rate limit/TPM detected. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                if "timeout" in err_msg and attempt == 1:
                    tqdm.write(f"⚠️ Timeout detected, retrying with top_k=3 for Q{idx+1}")
                    old_top_k = top_k
                    top_k = 3
                    try:
                        answer, _ = ask_question_groq(question, top_k=3)
                    except:
                        pass
                    top_k = old_top_k
                    continue

                tqdm.write(f"⚠️ Attempt {attempt} failed: {e}")
                time.sleep(2)
                if attempt == retry_limit:
                    df_out.at[idx, "response"] = "ERROR"
                    df_out.at[idx, "retrieval_time_sec"] = 0.0
                    df_out.at[idx, "response_time_sec"] = 0.0
                    df_out.at[idx, "semantic_similarity"] = 0.0
                    df_out.at[idx, "judge_score"] = 0.0
                    df_out.at[idx, "final_score"] = 0.0
                    df_out.at[idx, "status"] = "error"
                    tqdm.write(f"❌ Q{idx+1}: Failed after {retry_limit} attempts.")

        if (idx + 1) % save_interval == 0 or idx == total - 1:
            try:
                if output_file.endswith(".xlsx"):
                    df_out.to_excel(output_file, index=False)
                else:
                    df_out.to_csv(output_file, index=False)
                tqdm.write(f"💾 Progress saved at question {idx+1}.")
            except Exception as e:
                tqdm.write(f"❌ Error saving progress: {e}")

    tqdm.write("\n📊 Calculating averages for analysis...")

    df_success = df_out[df_out["status"] == "success"].copy()

    if len(df_success) == 0:
        avg_dict = {
            "retrieval_time_sec": 0.0,
            "response_time_sec": 0.0,
            "semantic_similarity": 0.0,
            "judge_score": 0.0,
            "final_score": 0.0
        }
    else:
        numeric_cols = ["retrieval_time_sec", "response_time_sec", "semantic_similarity", "judge_score", "final_score"]
        avg_dict = {}
        for col in numeric_cols:
            if col in df_success.columns:
                values = pd.to_numeric(df_success[col], errors='coerce')
                avg_dict[col] = values.mean() if not values.isna().all() else 0.0
            else:
                avg_dict[col] = 0.0

    total_questions = len(df_out)
    successful = len(df_out[df_out["status"] == "success"])
    not_found = len(df_out[df_out["status"] == "not_found"])
    errors = len(df_out[df_out["status"] == "error"])

    update_analysis_file(
        py_file=PY_FILE,
        output_file=output_file,
        input_file=input_file,
        model_embedding=MODEL_EMBEDDING_NAME,
        model_answer=MODEL_ANSWER,
        avg_retrieval_time=avg_dict.get("retrieval_time_sec", 0.0),
        avg_response_time=avg_dict.get("response_time_sec", 0.0),
        avg_semantic_sim=avg_dict.get("semantic_similarity", 0.0),
        avg_judge_score=avg_dict.get("judge_score", 0.0),
        avg_final_score=avg_dict.get("final_score", 0.0),
        successful_count=successful,
        total_questions=total_questions,
        not_found_count=not_found,
        error_count=errors
    )

    tqdm.write("💾 Saving final output file...")
    try:
        if output_file.endswith(".xlsx"):
            df_out.to_excel(output_file, index=False)
        else:
            df_out.to_csv(output_file, index=False)
        tqdm.write(f"✅ Final file saved to: {output_file}")
    except Exception as e:
        tqdm.write(f"❌ Error saving final file: {e}")

    tqdm.write("📊 Analysis summary added to 'AnalysisAll.xlsx'.")

# =========================
#  اجرا
# =========================

if __name__ == "__main__":
    run_batch_questions(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        top_k=10,
        retry_limit=7,
        save_interval=5
    )
