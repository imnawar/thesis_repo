import random
import pandas as pd
from openai import OpenAI

client = OpenAI()

# ==============================
# CONFIG
# ==============================

TOTAL_NEWS = 500
BATCH_SIZE = 50
NUM_BATCHES = TOTAL_NEWS // BATCH_SIZE

TOPICS = [
    "سياسة", "اقتصاد", "رياضة", "صحة", "تعليم",
    "بيئة", "تقنية", "ثقافة", "حوادث", "طاقة",
    "مجتمع", "قضايا اجتماعية"
]

ARAB_COUNTRIES = [
    "السعودية", "مصر", "الإمارات", "قطر", "الكويت",
    "البحرين", "عُمان", "الأردن", "المغرب", "تونس",
    "الجزائر", "لبنان", "العراق"
]

# ==============================
# PROMPT BUILDER
# ==============================

def build_few_shot_prompt(examples, topic, country):
    prompt = f"""
أنت نموذج لتوليد أخبار عربية واقعية بأسلوب صحفي احترافي.

🎯 المهمة:
- توليد خبر جديد بالكامل (عنوان + وصف صورة)
- يجب أن يكون مرتبطاً بالسياق العربي

📌 القيود المهمة:
- المجال: {topic}
- الدولة أو المنطقة: {country}

⚠️ قواعد صارمة:
- لا تكرر أي فكرة من الأمثلة
- لا تستخدم نفس الأسماء أو الأحداث أو الأرقام
- يجب أن يكون الخبر جديداً تماماً وغير مشابه لأي مثال
- يجب تنويع المواضيع بشكل كبير داخل نفس المجال
- تجنب التركيز على الذكاء الاصطناعي أو الإمارات بشكل متكرر
- اجعل الأخبار واقعية لكن متنوعة جغرافياً داخل العالم العربي

---

أمثلة للتعلم (الأسلوب فقط):

"""

    # shuffle examples to avoid pattern memorization
    examples = examples.sample(frac=1)

    for i, (_, row) in enumerate(examples.iterrows()):
        prompt += f"""
مثال {i+1}:
عنوان: {row['title']}
وصف الصورة: {row['caption']}
"""

    prompt += """

---

الآن قم بتوليد خبر جديد مختلف تماماً:

عنوان:
وصف الصورة:
"""

    return prompt


# ==============================
# GENERATION FUNCTION
# ==============================

def generate_batch(examples, batch_size):
    data = []

    for i in range(batch_size):
        topic = random.choice(TOPICS)
        country = random.choice(ARAB_COUNTRIES)

        prompt = build_few_shot_prompt(examples, topic, country)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت نموذج توليد أخبار عربية صحفية."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.95,   # 🔥 high randomness
            top_p=0.95
        )

        text = response.choices[0].message.content

        try:
            title = text.split("عنوان:")[1].split("وصف الصورة:")[0].strip()
            caption = text.split("وصف الصورة:")[1].strip()
        except:
            title = text
            caption = ""

        data.append({
            "title": title,
            "caption": caption,
            "topic": topic,
            "country": country
        })

    return pd.DataFrame(data)


# ==============================
# MAIN LOOP (500 samples)
# ==============================

all_data = []

for batch_idx in range(NUM_BATCHES):
    print(f"Generating batch {batch_idx + 1}/{NUM_BATCHES}...")

    # randomly sample few-shot examples per batch
    batch_examples = examples.sample(n=min(5, len(examples)))

    batch_df = generate_batch(batch_examples, BATCH_SIZE)
    all_data.append(batch_df)

final_df = pd.concat(all_data, ignore_index=True)

# save dataset
final_df.to_csv("fake_news_500_diverse.csv", index=False, encoding="utf-8-sig")

print("Done! Dataset saved.")
