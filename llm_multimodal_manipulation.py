import pandas as pd
from openai import OpenAI
import time
from difflib import SequenceMatcher

# 🔹 CONFIG
API_KEY = "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"
MODEL = "gpt-4.1-mini"
LIMIT = 20

client = OpenAI(api_key=API_KEY)


# 🔹 ================= PROMPTS =================

def build_title_prompt(title):
    return f"""
أنت محرر أخبار محترف.

المطلوب: تعديل العنوان ليصبح مضللاً ولكن واقعي جداً.

التعليمات:
- لا تغيّر كل الكلمات
- غيّر جزءاً واحداً فقط (رقم / جهة / سبب)
- حافظ على الأسلوب الصحفي

أمثلة:

حقيقي: ارتفاع أسعار النفط بنسبة 5%
مزيف: ارتفاع أسعار النفط بنسبة 25%

حقيقي: {title}
مزيف:
"""


def build_caption_prompt(caption):
    return f"""
أنت تكتب وصف صورة في خبر صحفي.

المطلوب: تعديل الوصف ليصبح مضللاً ويغير تفسير الصورة.

التعليمات:
- لا تغيّر العناصر الظاهرة في الصورة
- غيّر السياق فقط
- اجعله يبدو مقنعاً جداً

أمثلة:

حقيقي: تجمع لأشخاص خلال مهرجان
مزيف: تجمع لأشخاص خلال احتجاجات ضد الحكومة

حقيقي: {caption}
مزيف:
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
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Error:", e)
        return ""


# 🔹 ================= EVALUATION =================

def similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


def evaluate(real, fake):
    return {
        "similarity": similarity(real, fake),
        "length_real": len(str(real)),
        "length_fake": len(str(fake)),
        "length_diff": abs(len(str(real)) - len(str(fake)))
    }


# 🔹 ================= PIPELINE =================

def run_pipeline(input_csv):
    df = pd.read_csv(input_csv)

    results = []

    for i, row in df.iterrows():

        if i >= LIMIT:
            break

        real_title = str(row["title"])
        real_caption = str(row["caption"])

        # ---- Generate ----
        fake_title = generate_text(build_title_prompt(real_title))
        fake_caption = generate_text(build_caption_prompt(real_caption))

        # ---- Fallback ----
        if not fake_title or len(fake_title) < 5:
            fake_title = real_title

        if not fake_caption or len(fake_caption) < 5:
            fake_caption = real_caption

        # ---- Evaluate ----
        title_eval = evaluate(real_title, fake_title)
        caption_eval = evaluate(real_caption, fake_caption)

        results.append({
            "id": row["id"],

            "real_title": real_title,
            "fake_title": fake_title,
            "title_similarity": title_eval["similarity"],
            "title_length_diff": title_eval["length_diff"],

            "real_caption": real_caption,
            "fake_caption": fake_caption,
            "caption_similarity": caption_eval["similarity"],
            "caption_length_diff": caption_eval["length_diff"],

            "method": "llm_multimodal_manipulation"
        })

        print(f"Processed {i+1}")
        time.sleep(1)

    return pd.DataFrame(results)


# 🔹 ================= RUN =================

if __name__ == "__main__":
    input_file = "real_news.csv"

    df = run_pipeline(input_file)

    df.to_csv("llm_multimodal_eval.csv", index=False, encoding="utf-8-sig")

    print("\nDone! Saved to llm_multimodal_eval.csv")
