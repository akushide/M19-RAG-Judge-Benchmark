import os
M19_BASE = os.environ.get("M19_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
import os
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # <-- اضافه شده

# ========== ۱. تنظیمات آفلاین (قبل از هر چیز) ==========
os.environ["HF_HUB_OFFLINE"] = "1"  # جلوگیری از اتصال به اینترنت
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # غیرفعال‌سازی اخطار symlink

# ========== ۲. مسیرها ==========
JsonInputFile = os.path.join(M19_BASE, "parent_child_chunks.json")

#EmbedingOutputFile = os.path.join(M19_BASE, "M19RDBEmbeding_BGE_m3.json")
EmbedingOutputFile = os.path.join(M19_BASE, "M19RDBEmbeding_ML_E5_L.json")

#ModelPath = os.path.join(M19_BASE, "models", "bge-m3")
ModelPath = os.path.join(M19_BASE, "models", "multilingual-e5-large")

print(f"🔄 Loading from {ModelPath} ...")
model = SentenceTransformer(ModelPath, device="cpu")  # یا "cuda"
print("✅ The model successfully Loaded")
print("================================")
print("                                ")

# ========== ۴. خواندن فایل JSON ==========
with open(JsonInputFile, 'r', encoding='utf-8') as f:
    hierarchy = json.load(f)

nodes = hierarchy['nodes']
root_ids = hierarchy['root_ids']
print(f"🔹 Number of Nodes: {len(nodes)}")

# ========== ۵. تابع ساخت متن کامل ==========
def build_full_text(node, nodes):
    path = []
    current_id = node['id']
    while current_id is not None:
        current_node = nodes[current_id]
        if current_node.get('header'):
            path.insert(0, current_node['header'])
        current_id = current_node.get('parent_id')
    prefix = ' > '.join(path)
    content = node['content']
    if prefix and content:
        return f"{prefix}\n\n{content}"
    elif prefix:
        return prefix
    else:
        return content or ""

# ========== ۶. تولید Embedding ==========
embeddings = {}
total = len(nodes)
# استفاده از tqdm برای نمایش پیشرفت
for node_id, node in tqdm(nodes.items(), desc="Nodes process", unit="Nodes"):
    full_text = build_full_text(node, nodes)
    if full_text.strip():
        embedding = model.encode(full_text, normalize_embeddings=True)
        embeddings[node_id] = {
            'embedding': embedding.tolist(),
            'text': full_text,
            'metadata': {
                'level': node['level'],
                'header': node['header'],
                'parent_id': node['parent_id'],
                'children_ids': node['children_ids']
            }
        }

# ========== ۷. ذخیره خروجی ==========
with open(EmbedingOutputFile, 'w', encoding='utf-8') as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=2)

print(f"✅ Embedding for {len(embeddings)} node produced and in '{EmbedingOutputFile}' saved.")