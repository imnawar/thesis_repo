import pandas as pd
import time
from openai import OpenAI

# 🔹 ================= CONFIG =================
API_KEY = "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"
MODEL = "gpt-4.1-mini"
LIMIT = 1500   # number of samples to process

client = OpenAI(api_key=API_KEY)


# 🔹 ================= PROMPT =================

def build_manipulation_prompt(title, caption):
    return f"""
أنت محرر أخبار عربي محترف.

المطلوب:
قم بتعديل الخبر التالي ليصبح مضللاً ولكن واقعي جداً.

⚠️ التعليمات:
- لا تعيد كتابة النص بالكامل
- غيّر فقط جزءاً بسيطاً (مثل رقم، جهة، أو سبب)
- حافظ على نفس الأسلوب الصحفي
- يجب أن يبقى العنوان والوصف متسقين (نفس الحدث)

---

العنوان:
{title}

وصف الصورة:
{caption}

---

أعد النتيجة بهذا الشكل فقط:

عنوان: ...
وصف الصورة: ...
"""


# 🔹 ================= LLM =================

def generate_text(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "أنت محرر أخبار عربي محترف"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4  # 🔴 low = controlled changes
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Error:", e)
        return ""


# 🔹 ================= PARSER =================

def parse_output(text):
    title = ""
    caption = ""

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("عنوان"):
            title = line.split(":")[-1].strip()

        elif line.startswith("وصف"):
            caption = line.split(":")[-1].strip()

    return title, caption


# 🔹 ================= VALIDATION =================

def is_valid(text):
    return text and len(text) > 5


# 🔹 ================= MAIN PIPELINE =================

def run_pipeline(df):
    results = []

    for i, row in df.iterrows():

        if i >= LIMIT:
            break

        real_title = str(row["title"])
        real_caption = str(row["caption"])

        prompt = build_manipulation_prompt(real_title, real_caption)

        output = generate_text(prompt)

        fake_title, fake_caption = parse_output(output)

        # 🔴 fallback if model fails
        if not is_valid(fake_title):
            fake_title = real_title

        if not is_valid(fake_caption):
            fake_caption = real_caption

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


# 🔹 ================= RUN =================

if __name__ == "__main__":

    df = pd.read_csv("real_news.csv")

    # optional cleaning
    df.columns = df.columns.str.strip()

    output_df = run_pipeline(df)

    output_df.to_csv(
        "llm_manipulated_news.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone! Saved to llm_manipulated_news.csv")
