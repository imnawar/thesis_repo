import pandas as pd
import os
import time
from google import genai
from google.genai import types

# =========================================================
# CONFIG
# =========================================================
API_KEY = 'AIzaSyCIP-5zQYbOGRpEu2eqyaWmlY7ihuZvTuY'

CSV_FILE = 'fake_news_500_diverse.csv'

# Single output folder
OUTPUT_DIR = 'generated_images'

# Updated CSV path
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'generated_image_dataset.csv')

MODEL_ID = 'gemini-3.1-flash-image-preview'

MAX_ATTEMPTS = 2
SLEEP_BETWEEN_REQUESTS = 1.5

MAX_ROWS = 100

client = genai.Client(api_key=API_KEY)

# =========================================================
# DATA SELECTION
# =========================================================


# =========================================================
# IMAGE STYLE LOGIC
# =========================================================
def get_logic_tier_arabic(title):

    t = str(title).lower()

    low_keywords = [
        'مظاهرات',
        'اشتباكات',
        'حادث',
        'انفجار',
        'حريق',
        'اعتقال',
        'شرطة',
        'شغب',
        'عاجل'
    ]

    high_keywords = [
        'تعليم',
        'مؤتمر',
        'قمة',
        'حكومة',
        'وزير',
        'جامعة',
        'سياسة',
        'افتتاح',
        'رسمي',
        'تعاون',
        'لقاء'
    ]

    if any(w in t for w in low_keywords):

        return (
            "Grainy breaking-news photography with motion blur.",
            "4:3",
            "1K",
            "low"
        )

    elif any(w in t for w in high_keywords):

        return (
            "Professional DSLR journalism photography.",
            "3:2",
            "1K",
            "high"
        )

    else:

        return (
            "Realistic journalistic news photography.",
            "16:9",
            "1K",
            "standard"
        )

# =========================================================
# PROMPT
# =========================================================
def build_prompt(fake_title, fake_caption):

    style, ratio, res, tier = get_logic_tier_arabic(fake_title)

    prompt = f"""
    {style}

    Create a realistic Arabic news photo.

    Fake News Headline:
    {fake_title}

    Fake News Caption:
    {fake_caption}

    Requirements:
    - Realistic journalistic photography
    - Natural lighting
    - News-reporting style
    - High realism
    - No text
    - No logos
    - No watermark
    - No typography
    """

    return prompt, ratio, res, tier

# =========================================================
# MAIN
# =========================================================
def run_automation():

    # Create ONE output folder only
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_FILE)

    # Select one fake sample per real title
    # df = select_one_fake_per_real(df)

    # Limit rows
    df = df.head(MAX_ROWS).copy()

    print(f"\nProcessing {len(df)} rows...\n")

    # Create columns if not exist
    if "generated_image_path" not in df.columns:
        df["generated_image_path"] = ""

    if "quality_tier" not in df.columns:
        df["quality_tier"] = ""

    # =====================================================
    # LOOP
    # =====================================================
    for index, row in df.iterrows():

        filename = f"fake_news_{index}.png"

        image_path = os.path.join(OUTPUT_DIR, filename)

        # =================================================
        # USE FAKE TITLE + FAKE CAPTION ONLY
        # =================================================
        fake_title = str(row.get("fake_title", "")).strip()
        fake_caption = str(row.get("fake_caption", "")).strip()

        prompt, ratio, res, tier = build_prompt(
            fake_title,
            fake_caption
        )

        print(f"[{index+1}] Generating ({tier})")

        success = False

        # =================================================
        # RETRIES
        # =================================================
        for attempt in range(MAX_ATTEMPTS):

            try:

                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=ratio,
                            image_size=res
                        )
                    )
                )

                if response.parts:

                    for part in response.parts:

                        if part.inline_data:

                            # Save directly to final folder
                            part.as_image().save(image_path)

                            success = True
                            break

                if success:
                    break

            except Exception as e:

                err = str(e)

                if "503" in err:

                    print("   ! 503 error -> skipping")
                    break

                elif "429" in err:

                    print("   ! 429 rate limit -> waiting 10s")
                    time.sleep(10)

                else:

                    print(f"   ! Error: {e}")
                    time.sleep(2)

        # =================================================
        # SAVE RESULT
        # =================================================
        if success:

            df.at[index, "generated_image_path"] = image_path
            df.at[index, "quality_tier"] = tier

            print("   ✓ Saved")

        else:

            df.at[index, "generated_image_path"] = ""
            df.at[index, "quality_tier"] = "failed"

            print("   ✗ Failed")

        # =================================================
        # SAVE CSV AFTER EACH IMAGE
        # =================================================
        df.to_csv(OUTPUT_CSV, index=False)

        print("   ✓ CSV updated")

        # =================================================
        # WAIT
        # =================================================
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("\n✅ DONE")

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    run_automation()
