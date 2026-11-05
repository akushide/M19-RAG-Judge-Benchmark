import os
M19_BASE = os.environ.get("M19_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
import json
import chromadb
from chromadb.config import Settings
import os

# ۱. خواندن فایل امبدینگ
Fpath=os.path.join(M19_BASE)

#Fmodel="BGE_m3"
Fmodel="ML_E5_L"
#Fmodel="MiniLM_L12_v2"

FIndexing = "ChromaDB"    # تغییر نام برای تشخیص

JsonPath=os.path.join(Fpath , f"M19RDBEmbeding_{Fmodel}.json")
#JsonPath=os.path.join(M19_BASE, "M19RDBEmbeding_MiniLM_L12_v2.json")
#JsonPath=os.path.join(M19_BASE, "M19RDBEmbeding_ML_E5_L.json")

ChromaDB_Path  = os.path.join(Fpath , f"M19RDBEmbeding_{Fmodel}_{FIndexing}_indexing")

# ========== ۲. خواندن فایل امبدینگ ==========
with open(JsonPath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# تشخیص ساختار داده
if isinstance(data, dict):
    items = list(data.items())
elif isinstance(data, list):
    items = [(item.get('id', f"node_{i}"), item) for i, item in enumerate(data)]
else:
    raise TypeError("ساختار فایل JSON نامشخص است.")

print(f"🔹 Total number of Nodes: {len(items)}")

# ========== ۳. تابع پاک‌سازی متادیتا ==========
def clean_metadata(meta):
    """تمیز کردن متادیتا برای سازگاری با ChromaDB"""
    cleaned = {}
    for key, value in meta.items():
        # اگر مقدار None بود، به رشته خالی تبدیل کن
        if value is None:
            cleaned[key] = ""
        # اگر مقدار از نوع‌های ساده بود، همان را نگه دار
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        # اگر مقدار لیست یا دیکشنری بود، به JSON رشته تبدیل کن
        elif isinstance(value, (list, dict)):
            cleaned[key] = json.dumps(value, ensure_ascii=False)
        # هر نوع دیگری را به رشته تبدیل کن
        else:
            cleaned[key] = str(value)
    return cleaned

# ========== ۴. استخراج داده‌ها و یکتا‌سازی شناسه‌ها ==========
node_ids = []
embeddings = []
texts = []
metadatas = []
seen_ids = set()
id_counter = 0

for node_id, value in items:
    # یکتا‌سازی شناسه
    if node_id in seen_ids:
        new_id = f"{node_id}_{id_counter}"
        id_counter += 1
        print(f"⚠️ Repeated Indexes: {node_id} -> to {new_id} have been changed.")
        node_id = new_id
    seen_ids.add(node_id)
    
    node_ids.append(node_id)
    embeddings.append(value['embedding'])
    texts.append(value['text'])
    
    # پاک‌سازی متادیتا
    raw_meta = value.get('metadata', {})
    clean_meta = clean_metadata(raw_meta)
    metadatas.append(clean_meta)

print(f"🔹 Vector dimension: {len(embeddings[0])}")
print(f"🔹 Metadatas sample: {metadatas[0]}")

# ========== ۵. ایجاد کلاینت ChromaDB ==========
os.makedirs(ChromaDB_Path, exist_ok=True)
client = chromadb.PersistentClient(
    path=ChromaDB_Path,
    settings=Settings(anonymized_telemetry=False)
)

collection_name = f"chunks_{Fmodel}_{FIndexing}"
try:
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

# ========== ۶. افزودن داده‌ها به صورت دسته‌ای ==========
batch_size = 100
total = len(node_ids)

for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    print(f"🔄 adding data: {start+1} to {end} from {total}")
    
    collection.add(
        ids=node_ids[start:end],
        embeddings=embeddings[start:end],
        metadatas=metadatas[start:end],
        documents=texts[start:end]
    )


print(f"✅ Data successfully in ChromaDB are saved.")
print(f"📁 Database Path: {ChromaDB_Path}")
print(f"🔢 Collection count: {collection.count()}")