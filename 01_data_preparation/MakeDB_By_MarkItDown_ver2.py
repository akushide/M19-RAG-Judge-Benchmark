import os
M19_BASE = os.environ.get("M19_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
from markitdown import MarkItDown
import os
import re

input_path = os.path.join(M19_BASE, "ebook.docx")
output_path = os.path.join(M19_BASE, "ebookword2md.md")

if not os.path.exists(input_path):
    print(f"خطا: فایل '{input_path}' پیدا نشد!")
    exit()

try:
    md = MarkItDown()
    result = md.convert(input_path)

    # ۱. پاک‌سازی کاراکترهای جهتی
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', result.text_content)

    lines = text.split('\n')
    new_lines = []

    for line in lines:
        # ۲. ★ مرحله‌ی پیش‌پردازش (نرمال‌سازی خط) ★
        # حذف علامت‌های پررنگ (**) از ابتدا و انتهای خط (اگر وجود داشتند)
        cleaned_line = line.strip()
        if cleaned_line.startswith('**') and cleaned_line.endswith('**'):
            cleaned_line = cleaned_line[2:-2]  # حذف ** اول و ** آخر
        elif cleaned_line.startswith('**'):
            cleaned_line = cleaned_line[2:]     # فقط اول
        elif cleaned_line.endswith('**'):
            cleaned_line = cleaned_line[:-2]    # فقط آخر

        # ۳. ★ یکسان‌سازی خط فاصله‌ها ★
        # تمام خط فاصله‌های خاص (‑ – —) را به خط فاصله معمولی (-) تبدیل می‌کنیم
        normalized_line = re.sub(r'[‑–—]', '-', cleaned_line)

        # ۴. تابع کمکی برای تبدیل اعداد فارسی به انگلیسی
        def normalize_numbers(s):
            persian_to_english = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
            return s.translate(persian_to_english)

        # ۵. ★ تشخیص الگوها (از عمیق به سطحی) ★
        
        # سطح ۴ (مثال: 22-1-1-19 یا 22-1-19-1)
        match_l4 = re.match(r'^\s*([0-9۰-۹]+\s*[-]\s*[0-9۰-۹]+\s*[-]\s*[0-9۰-۹]+\s*[-]\s*[0-9۰-۹]+)\s+(.*)', normalized_line)
        if match_l4:
            raw = match_l4.group(1)
            title = match_l4.group(2).strip()
            norm_num = normalize_numbers(raw)
            norm_num = re.sub(r'\s*[-]\s*', '-', norm_num)
            # اگر عدد ۱۹ در ابتدا یا انتهای شماره بود، هدر محسوب کن
            if norm_num.startswith('19-') or norm_num.endswith('-19'):
                new_lines.append(f"#### {norm_num} {title}")
                continue

        # سطح ۳ (مثال: 22-1-19)
        match_l3 = re.match(r'^\s*([0-9۰-۹]+\s*[-]\s*[0-9۰-۹]+\s*[-]\s*[0-9۰-۹]+)\s+(.*)', normalized_line)
        if match_l3:
            raw = match_l3.group(1)
            title = match_l3.group(2).strip()
            norm_num = normalize_numbers(raw)
            norm_num = re.sub(r'\s*[-]\s*', '-', norm_num)
            if norm_num.startswith('19-') or norm_num.endswith('-19'):
                new_lines.append(f"### {norm_num} {title}")
                continue

        # سطح ۲ (مثال: 22-19 یا 19-22)
        match_l2 = re.match(r'^\s*([0-9۰-۹]+\s*[-]\s*[0-9۰-۹]+)\s+(.*)', normalized_line)
        if match_l2:
            raw = match_l2.group(1)
            title = match_l2.group(2).strip()
            norm_num = normalize_numbers(raw)
            norm_num = re.sub(r'\s*[-]\s*', '-', norm_num)
            if norm_num.startswith('19-') or norm_num.endswith('-19'):
                new_lines.append(f"## {norm_num} {title}")
                continue

        # اگر هیچکدام نبود، خط اصلی (قبل از نرمال‌سازی) را اضافه کن تا ظاهر پررنگ حفظ شود
        new_lines.append(line)

    final_text = '\n'.join(new_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print("✅ تبدیل با موفقیت انجام شد! هدرها با پشتیبانی از ** و خط فاصله‌های ویژه تشخیص داده شدند.")

except Exception as e:
    print(f"❌ خطا در حین تبدیل: {e}")