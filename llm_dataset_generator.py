import pandas as pd
import time
from openai import OpenAI

# 🔹 ================= CONFIG =================
API_KEY = "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"
MODEL = "gpt-4.1-mini"
NUM_SAMPLES_TO_GENERATE = 500
NUM_EXAMPLES = 5

client = OpenAI(api_key=API_KEY)


# 🔹 ================= PROMPT BUILDER =================

def build_few_shot_prompt(examples):
    prompt = """
أنت نموذج لتوليد أخبار عربية واقعية بأسلوب يشبه الصحافة الإخبارية.

مهمتك:
- تعلم الأسلوب من الأمثلة
- ثم توليد خبر جديد (عنوان + وصف صورة)
- يجب أن يكون متسقاً (العنوان والوصف لنفس الحدث)

---

أمثلة:

"""

    # ✅ FIX HERE
    for i, (_, row) in enumerate(examples.iterrows()):
        prompt += f"""
مثال {i+1}:
عنوان: {row['title']}
وصف الصورة: {row['caption']}
"""

    prompt += """

---

الآن قم بتوليد خبر جديد:

عنوان: ...
وصف الصورة: ...
"""

    return prompt



# 🔹 ================= LLM CALL =================

def generate_text(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "أنت كاتب أخبار عربي محترف"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9  # diversity
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Error:", e)
        return ""


# 🔹 ================= PARSE OUTPUT =================

def parse_generated_output(text):
    title = ""
    caption = ""

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("عنوان"):
            title = line.split(":")[-1].strip()

        elif line.startswith("وصف"):
            caption = line.split(":")[-1].strip()

    return title, caption


# 🔹 ================= GENERATION PIPELINE =================

def generate_dataset(df):
    results = []

    for i in range(NUM_SAMPLES_TO_GENERATE):

        # 🔥 sample few-shot examples
        examples = df.sample(NUM_EXAMPLES)

        prompt = build_few_shot_prompt(examples)

        output = generate_text(prompt)

        fake_title, fake_caption = parse_generated_output(output)

        # 🔴 validation fallback
        if not fake_title or not fake_caption:
            print(f"Skipping sample {i+1} (empty output)")
            continue

        results.append({
            "fake_title": fake_title,
            "fake_caption": fake_caption,
            "method": "dataset_few_shot_generation"
        })

        print(f"Generated {i+1}/{NUM_SAMPLES_TO_GENERATE}")

        time.sleep(1)

    return pd.DataFrame(results)


# 🔹 ================= RUN =================

if __name__ == "__main__":

    df = pd.read_csv("real_news.csv")

    synthetic_df = generate_dataset(df)

    synthetic_df.to_csv(
        "llm_dataset_generated_news.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone! Dataset saved: llm_dataset_generated_news.csv")
