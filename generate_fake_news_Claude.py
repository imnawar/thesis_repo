import random
import time
import logging
import pandas as pd
import anthropic
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
        logging.FileHandler("generation_claude.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy third-party SDK logs (httpx request lines), same reasoning
# as the Gemini script — we only want our own progress/warning/error lines.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

# ==============================
# CONFIG
# ==============================
TOTAL_NEWS      = 1500
BATCH_SIZE      = 50
NUM_BATCHES     = TOTAL_NEWS // BATCH_SIZE
MAX_WORKERS     = 3          # reduced to avoid overwhelming the API with 429/overloaded errors
MAX_RETRIES     = 5          # more retries to handle transient overloaded errors
INPUT_CSV       = "real_news.csv"
OUTPUT_CSV      = "fake_news_1500_claude.csv"
CHECKPOINT_DIR  = Path("checkpoints_claude")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Using Sonnet 4.6 because it supports custom temperature/top_p, which we
# rely on below for generation diversity. Claude Sonnet 5 (model id
# "claude-sonnet-5") is available and is the newer flagship-tier Sonnet
# model, but it rejects non-default temperature/top_p values (400 error)
# since it uses adaptive thinking by default. Swap the string below if you
# decide diversity-via-temperature isn't essential for your use case.
CLAUDE_MODEL    = "claude-sonnet-4-6"

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

# Create a single shared client — the Anthropic SDK client is thread-safe.
# Reads ANTHROPIC_API_KEY automatically from environment.
client = anthropic.Anthropic()

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
            response = client.messages.create(
                model=CLAUDE_MODEL,
                system=SYSTEM_INSTRUCTION,
                max_tokens=2048,       # same headroom as the Gemini script
                temperature=0.95,      # Claude API rejects setting both temperature
                                       # and top_p together, so top_p is dropped
                                       # (it was 1.0 = no-op in the Gemini script anyway)
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            text           = response.content[0].text
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
    logger.info(f"Using Claude model: {CLAUDE_MODEL}")

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