import os
M19_BASE = os.environ.get("M19_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
import json
import re

chunks_json_path = os.path.join(M19_BASE, "M19RDB.md.chunks.json")
output_parentchild_json_path = os.path.join(M19_BASE, "parent_child_chunks.json")
output_parentchild_txt_path = os.path.join(M19_BASE, "parent_child_chunks.txt")


def export_hierarchy_to_txt(hierarchy_data, output_path):
    nodes = hierarchy_data['nodes']
    root_ids = hierarchy_data['root_ids']
    
    def write_node(node_id, level, file):
        node = nodes[node_id]
        indent = " " * (level * 4)
        header_mark = "#" * node['level'] if node['level'] > 0 else ""
        if header_mark:
            file.write(f"{indent}{header_mark} {node['header']}\n")
        else:
            file.write(f"{indent}[بدون هدر]\n")
        if node['content'].strip():
            for line in node['content'].split('\n'):
                file.write(f"{indent}  {line}\n")
            file.write("\n")
        for child_id in node['children_ids']:
            write_node(child_id, level + 1, file)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ساختار سلسله‌مراتبی چانک‌ها (والد-فرزند) - یکتا شده\n")
        f.write("=" * 80 + "\n\n")
        for root_id in root_ids:
            write_node(root_id, 0, f)
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"✅ فایل متنی در '{output_path}' ذخیره شد.")

def parse_headers(header_line):
    """خط هدرها را به لیست هدرها تبدیل می‌کند."""
    return [h.strip() for h in header_line.split('>') if h.strip()]

def build_parent_child_chunks(flat_chunks):
    """
    چانک‌های تخت را به ساختار والد-فرزند تبدیل می‌کند.
    ★ گره‌های تکراری را تشخیص داده و از همان گره استفاده می‌کند.
    """
    nodes = {}
    root_ids = []
    node_counter = 0
    header_to_node_id = {}  # کش: نگاشت هدر به شناسه گره

    for chunk in flat_chunks:
        text = chunk['text']
        if not text.strip():
            continue

        # جداسازی هدر و بدنه
        if '\n\n' in text:
            header_line, body = text.split('\n\n', 1)
        else:
            lines = text.split('\n')
            header_line = lines[0] if lines else ''
            body = '\n'.join(lines[1:]) if len(lines) > 1 else ''

        headers = parse_headers(header_line)
        if not headers:
            node_id = f"node_{node_counter}"
            node_counter += 1
            nodes[node_id] = {
                "id": node_id,
                "level": 0,
                "header": None,
                "content": text,
                "children_ids": [],
                "parent_id": None
            }
            root_ids.append(node_id)
            continue

        # پردازش سلسله‌مراتب
        parent_id = None
        for level_index, header in enumerate(headers, start=1):
            # ★ کلید یکتا بر اساس ترکیب والد و هدر جاری
            parent_key = parent_id if parent_id else "root"
            unique_key = f"{parent_key}::{header}"

            # اگر این هدر قبلاً در این سطح و با همین والد دیده شده، از همان گره استفاده کن
            if unique_key in header_to_node_id:
                node_id = header_to_node_id[unique_key]
            else:
                # گره جدید بساز
                node_id = f"node_{node_counter}"
                node_counter += 1
                content = body if level_index == len(headers) else ""
                node = {
                    "id": node_id,
                    "level": level_index,
                    "header": header,
                    "content": content,
                    "children_ids": [],
                    "parent_id": parent_id
                }
                nodes[node_id] = node
                header_to_node_id[unique_key] = node_id

                # اتصال به والد (فقط برای گره‌های جدید)
                if parent_id is not None:
                    if node_id not in nodes[parent_id]["children_ids"]:
                        nodes[parent_id]["children_ids"].append(node_id)
                else:
                    if node_id not in root_ids:
                        root_ids.append(node_id)

            # به‌روزرسانی parent_id برای سطح بعدی
            parent_id = node_id

    return {"nodes": nodes, "root_ids": root_ids}


# ==================== اجرا ====================

with open(chunks_json_path, 'r', encoding='utf-8') as f:
    flat_chunks = json.load(f)

result = build_parent_child_chunks(flat_chunks)

with open(output_parentchild_json_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ گره‌های یکتا: {len(result['nodes'])}")
print(f"✅ ریشه‌ها: {len(result['root_ids'])}")


with open(output_parentchild_json_path, 'r', encoding='utf-8') as f:
    hierarchy_data = json.load(f)

# ۲. تبدیل به فایل متنی
export_hierarchy_to_txt(hierarchy_data, output_parentchild_txt_path)