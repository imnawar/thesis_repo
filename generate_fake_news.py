import random
import time
import logging
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
# ================= CONFIG =================

API_KEY = "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"

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("generation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# CONFIG
# ==============================
TOTAL_NEWS      = 1500
BATCH_SIZE      = 50
NUM_BATCHES     = TOTAL_NEWS // BATCH_SIZE
MAX_WORKERS     = 5          # parallel API calls per batch
MAX_RETRIES     = 3          # retries per failed API call
INPUT_CSV       = "real_news.csv"
OUTPUT_CSV      = "fake_news_500_diverse_gpt4_5_v.csv"
CHECKPOINT_DIR  = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

TOPICS = [
    "سياسة", "اقتصاد", "رياضة", "صحة", "تعليم",
    "بيئة", "تقنية", "ثقافة", "حوادث", "طاقة",
    "مجتمع", "قضايا اجتماعية"
]
ARAB_COUNTRIES = [
    "السعودية", "مصر", "الإمارات", "قطر", "الكويت",
    "البحرين", "عُمان", "الأردن", "المغرب", "تونس",
    "الجزائر", "لبنان", "العراق", "ليبيا", "سوريا",
    "اليمن", "فلسطين", "الصومال", "جيبوتي",
    "موريتانيا", "جزر القمر", "السودان"
]

client = OpenAI(api_key=API_KEY)

# ==============================
# PROMPT BUILDER
# ==============================
def build_few_shot_prompt(examples: pd.DataFrame, topic: str, country: str) -> str:
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
    for i, (_, row) in enumerate(examples.sample(frac=1).iterrows()):
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
# PARSE RESPONSE
# ==============================
def parse_response(text: str, item_idx: int) -> tuple[str, str]:
    try:
        title   = text.split("عنوان:")[1].split("وصف الصورة:")[0].strip()
        caption = text.split("وصف الصورة:")[1].strip()
        return title, caption
    except (IndexError, ValueError) as e:
        logger.warning(f"⚠️  Parse failed for item {item_idx}: {e} | Raw: {text[:200]!r}")
        return text.strip(), ""

# ==============================
# SINGLE ITEM GENERATION (with retry)
# ==============================
def generate_single(args: tuple) -> dict | None:
    item_idx, examples, topic, country = args
    prompt = build_few_shot_prompt(examples, topic, country)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "أنت نموذج توليد أخبار عربية صحفية."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.95,
                top_p=1.0,          # avoid combining both; let temperature drive diversity
            )
            text           = response.choices[0].message.content
            title, caption = parse_response(text, item_idx)
            return {"title": title, "caption": caption, "topic": topic, "country": country}

        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"API error on item {item_idx} (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s…")
            time.sleep(wait)

    logger.error(f"❌ Item {item_idx} failed after {MAX_RETRIES} attempts. Skipping.")
    return None

# ==============================
# BATCH GENERATION (parallel)
# ==============================
def generate_batch(
    batch_idx: int,
    examples: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:

    # build task args for this batch
    tasks = [
        (batch_idx * batch_size + i, examples.sample(n=min(5, len(examples))), random.choice(TOPICS), random.choice(ARAB_COUNTRIES))
        for i in range(batch_size)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_single, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)

# ==============================
# MAIN LOOP
# ==============================
def main():
    # Load examples ONCE before the loop
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    examples = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(examples)} real news examples from '{INPUT_CSV}'.")

    all_data = []

    for batch_idx in range(NUM_BATCHES):
        checkpoint_path = CHECKPOINT_DIR / f"batch_{batch_idx + 1}.csv"

        # Resume from checkpoint if it already exists
        if checkpoint_path.exists():
            logger.info(f"Batch {batch_idx + 1}/{NUM_BATCHES}: checkpoint found, skipping generation.")
            batch_df = pd.read_csv(checkpoint_path, encoding="utf-8-sig")
        else:
            logger.info(f"Generating batch {batch_idx + 1}/{NUM_BATCHES}…")
            batch_df = generate_batch(batch_idx, examples, BATCH_SIZE)

            # Save checkpoint immediately after each batch
            batch_df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
            logger.info(f"✅ Batch {batch_idx + 1} saved ({len(batch_df)} items) → {checkpoint_path}")

        all_data.append(batch_df)

    # Combine and deduplicate
    final_df = pd.concat(all_data, ignore_index=True)
    before   = len(final_df)
    final_df = final_df.drop_duplicates(subset=["title"])
    after    = len(final_df)

    if before != after:
        logger.info(f"🧹 Removed {before - after} duplicate titles. {after} unique samples remain.")

    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"🎉 Done! {after} samples saved → '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
