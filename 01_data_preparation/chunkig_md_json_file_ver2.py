import os
M19_BASE = os.environ.get("M19_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
import re
import json  # برای نمایش زیبای خروجی
import os

import re

def chunkig_md_json_file(file_text):
    """
    یک فایل مارک‌داون را به قطعات (چانک) تقسیم می‌کند.
    
    ★ ویژگی جدید: شروع بلوک ```json به عنوان یک چانک جدید شناخته می‌شود.
    این کار باعث می‌شود هر جدول (JSON) در یک چانک مجزا قرار گیرد
    در حالی که سلسله‌مراتب هدرهای والد خود را حفظ می‌کند.
    
    پارامترها:
        file_text (str): محتوای کامل فایل مارک‌داون به صورت رشته

    بازگشت:
        list: لیستی از دیکشنری‌ها با کلیدهای 'id' و 'text'
    """
    lines = file_text.split('\n')
    chunks = []
    current_chunk_lines = []
    header_stack = []          # سلسله‌مراتب هدرهای فعال
    inside_code_block = False  # برای مدیریت کلی بلوک‌های کد

    # ---------- توابع کمکی ----------
    def get_header_level(line):
        """سطح هدر را بر اساس # و * تشخیص می‌دهد (همانند قبل)"""
        stripped = line.strip()
        
        match_hash = re.match(r'^(#{1,4})\s+', stripped)
        if match_hash:
            return len(match_hash.group(1))
        
        match_star = re.match(r'^\*{1,2}\s+', stripped)
        if match_star:
            return 6 if stripped.startswith('**') else 5
        
        return 0

    def build_context_prefix(stack):
        return " > ".join(stack)

    def save_current_chunk():
        """قطعه‌ی فعلی را به لیست اضافه می‌کند و آن را خالی می‌کند"""
        nonlocal current_chunk_lines
        if current_chunk_lines:
            chunk_body = '\n'.join(current_chunk_lines)
            context_prefix = build_context_prefix(header_stack)
            final_chunk = f"{context_prefix}\n\n{chunk_body}" if context_prefix else chunk_body
            chunks.append(final_chunk)
            current_chunk_lines = []

    # ---------- حلقه‌ی اصلی ----------
    for line in lines:
        stripped = line.strip()
        
        # ★ بخش ۱: تشخیص شروع بلوک JSON (جدول) ★
        # اگر داخل بلوک کد نیستیم و خط با ```json شروع شد
        if not inside_code_block and stripped.startswith('```json'):
            # چانک قبلی را ذخیره کن (متن قبل از جدول)
            save_current_chunk()
            
            # شروع بلوک کد را علامت‌گذاری کن
            inside_code_block = True
            # خط ```json را به چانک جدید اضافه کن
            current_chunk_lines.append(line)
            continue  # ادامه می‌دهیم تا خطوط بعدی پردازش شوند

        # ★ بخش ۲: تشخیص پایان بلوک کد ★
        if inside_code_block and stripped.startswith('```'):
            # خط ``` را به چانک فعلی اضافه کن
            current_chunk_lines.append(line)
            inside_code_block = False
            
            # ★ (اختیاری) اگر می‌خواهید بعد از جدول حتماً چانک جدیدی شروع شود،
            # می‌توانید اینجا هم save_current_chunk() را صدا بزنید.
            # اما برای اینکه توضیحات بلافاصله‌ی بعد از جدول به همان چانک بچسبد،
            # اینجا save نمی‌کنیم. (در غیر این صورت، اگر بعد از جدول توضیح داشت، جدا می‌شد)
            continue

        # ★ بخش ۳: تشخیص هدرهای معمولی (#, ##, ###, ####, *, **) ★
        is_header = False
        header_level = 0
        if not inside_code_block:
            header_level = get_header_level(line)
            if header_level > 0:
                is_header = True

        if is_header:
            # چانک قبلی را ذخیره کن
            save_current_chunk()
            
            # به‌روزرسانی پشته‌ی سلسله‌مراتب
            while header_stack and get_header_level(header_stack[-1]) >= header_level:
                header_stack.pop()
            header_stack.append(line.strip())

        # ★ بخش ۴: اضافه کردن خط جاری به چانک فعلی ★
        current_chunk_lines.append(line)

    # ---- ذخیره‌سازی آخرین چانک ----
    save_current_chunk()

    # اختصاص شناسه عددی
    return [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]


# ۱. مسیر فایل مارک‌داون خروجی (همان فایلی که از ورد تبدیل کردید)
md_file_path = os.path.join(M19_BASE, "M19RDB.md")
output_dir = os.path.join(M19_BASE)
base_name = "M19RDB.md"
output_chunks_path = os.path.join(output_dir, f"{base_name}.chunks.json")

# ۲. خواندن محتوای فایل
with open(md_file_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# ۳. اجرای تابع چانک‌بندی
chunks = chunkig_md_json_file(md_content)

# ۴. نمایش تعداد چانک‌ها و نمونه‌ی اول
print(f"✅ تعداد کل چانک‌ها: {len(chunks)}")
print("\n--- نمونه چانک اول ---")
print(chunks[0]['text'][:300])  # ۳۰۰ کاراکتر اول

# ۵. (اختیاری) ذخیره چانک‌ها در یک فایل JSON برای استفاده در RAG
with open(output_chunks_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ فایل چانک‌ها در '{output_chunks_path}' ذخیره شد.")



chunks_json_path = os.path.join(M19_BASE, "M19RDB.md.chunks.json")
output_txt_path = os.path.join(M19_BASE, "M19RDB_structured.txt")

with open(chunks_json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(output_txt_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(f"{'='*50}\n")          # خط جداکننده
        f.write(f"Chunk ID: {chunk['id']}\n")
        f.write(f"{'-'*50}\n")
        f.write(chunk['text'])
        f.write(f"\n\n")                # فاصله بین چانک‌ها

print(f"✅ فایل ساختاریافته در '{output_txt_path}' ذخیره شد.")