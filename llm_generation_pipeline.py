import pandas as pd
from openai import OpenAI
import time

# 🔹 ========== CONFIG ==========
API_KEY = "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"   # 🔴 PUT YOUR KEY HERE
MODEL = "gpt-4.1-mini"     # fast + good quality
OUTPUT_FILE = "fake_news_generated.csv"
LIMIT = 20  # 🔴 start small to test

client = OpenAI(api_key=API_KEY)


# 🔹 ========== PROMPTS ==========

def build_title_prompt(title):
    return f"""
أنت كاتب أخبار محترف. مهمتك إعادة صياغة العنوان التالي ليصبح خبراً مضللاً ولكن يبدو واقعياً.

التعليمات:
- حافظ على الأسلوب الصحفي
- لا تبالغ أو تجعل الخبر خيالياً
- غيّر المعنى ليصبح مضللاً
- اجعله قريباً في الطول من العنوان الأصلي

أمثلة:

حقيقي: ارتفاع أسعار النفط بنسبة 5%
مزيف: ارتفاع أسعار النفط بسبب أزمة اقتصادية عالمية مفاجئة

حقيقي: إطلاق مشروع سكني جديد في الرياض
مزيف: إطلاق مشروع سكني متعثر يثير استياء المواطنين

حقيقي: {title}
مزيف:
"""


def build_caption_prompt(caption):
    return f"""
أنت تكتب وصفاً لصورة في خبر صحفي.

المطلوب: تعديل الوصف ليصبح مضللاً ويغير سياق الصورة مع الحفاظ على الواقعية.

التعليمات:
- لا تجعل الوصف خيالياً
- غيّر التفسير فقط
- اجعله يبدو مقنعاً

أمثلة:

حقيقي: تجمع لأشخاص خلال مهرجان
مزيف: تجمع لأشخاص خلال احتجاجات ضد الحكومة

حقيقي: {caption}
مزيف:
"""


# 🔹 ========== LLM CALL ==========

def generate_text(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "أنت كاتب أخبار عربي محترف"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Error:", e)
        return ""


# 🔹 ========== VALIDATION ==========

def is_valid(text):
    if not text:
        return False
    if len(text) < 10:
        return False
    return True


# 🔹 ========== GENERATION FUNCTION ==========

def generate_fake_sample(title, caption):
    title_prompt = build_title_prompt(title)
    caption_prompt = build_caption_prompt(caption)

    fake_title = generate_text(title_prompt)
    fake_caption = generate_text(caption_prompt)

    if not is_valid(fake_title):
        fake_title = title  # fallback

    if not is_valid(fake_caption):
        fake_caption = caption

    return fake_title, fake_caption


# 🔹 ========== MAIN PIPELINE ==========

def run_generation(input_csv):
    df = pd.read_csv(input_csv)

    results = []

    for i, row in df.iterrows():

        if LIMIT and i >= LIMIT:
            break

        title = row["title"]
        caption = row["caption"]

        fake_title, fake_caption = generate_fake_sample(title, caption)

        results.append({
            "id": row["id"],
            "real_title": title,
            "real_caption": caption,
            "fake_title": fake_title,
            "fake_caption": fake_caption,
            "method": "llm_generated",
            "label": "fake"
        })

        print(f"Processed {i+1}")

        # 🔴 avoid rate limits
        time.sleep(1)

    return pd.DataFrame(results)


# 🔹 ========== RUN ==========

if __name__ == "__main__":
    input_file = "real_news.csv"

    fake_df = run_generation(input_file)

    fake_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Done! Saved to:", OUTPUT_FILE)
