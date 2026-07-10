import random
import time
import logging
import pandas as pd
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file from the same directory as this script
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("generation_gemini.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy third-party SDK logs (httpx request lines + AFC notices).
# These are emitted at INFO level by the google-genai SDK and its HTTP
# client on every single API call, which floods the log when running
# thousands of generations across multiple threads. We only want OUR
# own progress/warning/error messages to show up.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# ==============================
# CONFIG
# ==============================
TOTAL_NEWS      = 1500
BATCH_SIZE      = 50
NUM_BATCHES     = TOTAL_NEWS // BATCH_SIZE
MAX_WORKERS     = 3          # reduced to avoid overwhelming the API with 503s
MAX_RETRIES     = 5          # more retries to handle 503 demand spikes
INPUT_CSV       = "real_news.csv"
OUTPUT_CSV      = "fake_news_1500_gemini.csv"
CHECKPOINT_DIR  = Path("checkpoints_gemini")
CHECKPOINT_DIR.mkdir(exist_ok=True)

GEMINI_MODEL    = "gemini-2.5-flash"   # fast & cost-effective; swap to "gemini-2.5-pro" for higher quality

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

SYSTEM_INSTRUCTION = "أنت نموذج توليد أخبار عربية صحفية."

# Create a single shared client — the new google-genai SDK is thread-safe
# Reads GOOGLE_API_KEY or GEMINI_API_KEY automatically from environment
client = genai.Client()

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

    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.95,
        top_p=1.0,
        max_output_tokens=2048,  # increased to avoid truncated Arabic responses
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=generation_config,
            )
            text           = response.text
            title, caption = parse_response(text, item_idx)
            return {"title": title, "caption": caption, "topic": topic, "country": country}

        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 2)  # jitter to avoid retry storms
            logger.warning(
                f"API error on item {item_idx} (attempt {attempt}/{MAX_RETRIES}): {e}. "
                f"Retrying in {wait:.1f}s…"
            )
            time.sleep(wait)

    logger.error(f"❌ Item {item_idx} failed after {MAX_RETRIES} attempts. Skipping.")
    return None

# ==============================
# BATCH GENERATION (parallel)
# ==============================
def generate_batch(batch_idx: int, examples: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    tasks = [
        (
            batch_idx * batch_size + i,
            examples.sample(n=min(5, len(examples))),
            random.choice(TOPICS),
            random.choice(ARAB_COUNTRIES),
        )
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
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    examples = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(examples)} real news examples from '{INPUT_CSV}'.")
    logger.info(f"Using Gemini model: {GEMINI_MODEL}")

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