import pandas as pd
import time
import re
from difflib import SequenceMatcher
from openai import OpenAI

# ================= CONFIG =================

API_KEY = "YOUR_API_KEY"
MODEL = "gpt-4.1-mini"

LIMIT = 1500
MAX_RETRIES = 3

client = OpenAI(api_key=API_KEY)

# ================= TEXT CLEANING =================

def normalize_text(text):
    text = str(text).strip().lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text


# ================= SIMILARITY =================

def similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def too_similar(real_text, fake_text, threshold=0.75):
    return similarity(real_text, fake_text) >= threshold


# ================= PROMPT =================

def build_manipulation_prompt(title, caption):

    return f"""
أنت محرر أخبار متخصص في إنشاء أخبار مضللة واقعية لأغراض بحثية.

المطلوب:
إعادة صياغة العنوان ووصف الصورة ليصبحا مضللين لكن مقنعين وواقعيين.

تعليمات مهمة جداً:

1- يجب أن يكون العنوان الجديد مختلفاً بوضوح عن العنوان الأصلي.
2- يجب أن يكون وصف الصورة الجديد مختلفاً بوضوح عن الوصف الأصلي.
3- لا تنسخ أي جملة كاملة من النص الأصلي.
4- غيّر الحدث أو السبب أو الجهة أو النتيجة بطريقة تجعل الخبر مضللاً.
5- حافظ على الترابط الكامل بين العنوان والوصف.
6- استخدم أسلوباً صحفياً طبيعياً واحترافياً.
7- لا تكرر نفس الكلمات الرئيسية قدر الإمكان.
8- يجب أن تبدو النسخة المزيفة كخبر مستقل وليس تعديلًا بسيطًا.

العنوان الأصلي:
{title}

وصف الصورة الأصلي:
{caption}

أعد النتيجة بهذا الشكل فقط:

عنوان: ...
وصف الصورة: ...
"""


# ================= GENERATION =================

def generate_text(prompt):

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content":
                    "أنت محرر أخبار عربي محترف ومتخصص في إعادة صياغة الأخبار بشكل مضلل لكن واقعي."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.9
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Generation Error:", e)
        return ""


# ================= PARSER =================

def parse_output(text):

    title = ""
    caption = ""

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if line.startswith("عنوان"):
            parts = line.split(":", 1)

            if len(parts) > 1:
                title = parts[1].strip()

        elif line.startswith("وصف"):
            parts = line.split(":", 1)

            if len(parts) > 1:
                caption = parts[1].strip()

    return title, caption


# ================= VALIDATION =================

def is_valid(text):
    return text and len(text.strip()) > 10


def validate_generation(
    real_title,
    fake_title,
    real_caption,
    fake_caption
):

    # empty check
    if not is_valid(fake_title):
        return False

    if not is_valid(fake_caption):
        return False

    # exact duplication
    if normalize_text(real_title) == normalize_text(fake_title):
        return False

    if normalize_text(real_caption) == normalize_text(fake_caption):
        return False

    # similarity check
    if too_similar(real_title, fake_title):
        return False

    if too_similar(real_caption, fake_caption):
        return False

    return True


# ================= SINGLE SAMPLE =================

def generate_fake_pair(real_title, real_caption):

    for attempt in range(MAX_RETRIES):

        prompt = build_manipulation_prompt(
            real_title,
            real_caption
        )

        output = generate_text(prompt)

        fake_title, fake_caption = parse_output(output)

        valid = validate_generation(
            real_title,
            fake_title,
            real_caption,
            fake_caption
        )

        if valid:
            return fake_title, fake_caption

        print(f"Retrying generation... attempt {attempt+1}")

    # final fallback
    return (
        "[FAILED_GENERATION]",
        "[FAILED_GENERATION]"
    )


# ================= MAIN PIPELINE =================

def run_pipeline(df):

    results = []

    for i, row in df.iterrows():

        if i >= LIMIT:
            break

        real_title = str(row["title"])
        real_caption = str(row["caption"])

        fake_title, fake_caption = generate_fake_pair(
            real_title,
            real_caption
        )

        results.append({
            "id": row["id"],
            "real_title": real_title,
            "fake_title": fake_title,
            "real_caption": real_caption,
            "fake_caption": fake_caption,
            "method": "llm_manipulation"
        })

        print(f"Processed {i+1}/{LIMIT}")

        time.sleep(1)

    return pd.DataFrame(results)


# ================= RUN =================

if __name__ == "__main__":

    df = pd.read_csv("real_news.csv")

    df.columns = df.columns.str.strip()

    output_df = run_pipeline(df)

    output_df.to_csv(
        "llm_manipulated_news.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone! Saved to llm_manipulated_news.csv")
