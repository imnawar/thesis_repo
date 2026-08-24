"""
Fake-news image generation pipeline (Gemini image model).

Generates a synthetic image for each row of a CSV containing fake news
titles/captions, tagging each with a "quality tier" derived from Arabic
keyword heuristics. Designed to be safely re-run: already-generated rows
are skipped, and progress is checkpointed to disk after every image.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python generate_images.py --csv fake_news_500_diverse.csv --max-rows 100
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable


# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    csv_file: str = "fake_news_merged.csv"
    output_dir: str = "generated_images"
    model_id: str = "gemini-3.1-flash-image-preview"
    max_attempts: int = 3
    base_sleep: float = 1.5          # sleep between successful requests
    max_rows: int = 4431
    log_file: str = "generation.log"

    @property
    def output_csv(self) -> str:
        return os.path.join(self.output_dir, "generated_image_dataset.csv")


def build_client() -> genai.Client:
    """Load the API key from the environment rather than hardcoding it.

    Set it with:  export GEMINI_API_KEY="your-key-here"
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Run: export GEMINI_API_KEY='your-key-here'"
        )
    return genai.Client(api_key=api_key)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("image_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# =========================================================
# IMAGE STYLE LOGIC
# =========================================================
LOW_KEYWORDS = [
    "مظاهرات", "اشتباكات", "حادث", "انفجار",
    "حريق", "اعتقال", "شرطة", "شغب", "عاجل",
]

HIGH_KEYWORDS = [
    "تعليم", "مؤتمر", "قمة", "حكومة", "وزير",
    "جامعة", "سياسة", "افتتاح", "رسمي", "تعاون", "لقاء",
]


def get_logic_tier_arabic(title: str) -> tuple[str, str, str, str]:
    """Return (style_description, aspect_ratio, resolution, tier) for a title."""
    t = str(title).lower()

    if any(w in t for w in LOW_KEYWORDS):
        return ("Grainy breaking-news photography with motion blur.", "4:3", "1K", "low")

    if any(w in t for w in HIGH_KEYWORDS):
        return ("Professional DSLR journalism photography.", "3:2", "1K", "high")

    return ("Realistic journalistic news photography.", "16:9", "1K", "standard")


# =========================================================
# PROMPT
# =========================================================
def build_prompt(fake_title: str, fake_caption: str) -> tuple[str, str, str, str]:
    style, ratio, res, tier = get_logic_tier_arabic(fake_title)

    prompt = f"""{style}

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
# GENERATION (single row, with retry + backoff)
# =========================================================
def generate_one_image(
    client: genai.Client,
    cfg: Config,
    prompt: str,
    ratio: str,
    res: str,
    image_path: str,
    logger: logging.Logger,
) -> bool:
    """Attempt to generate and save one image. Returns True on success."""
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=cfg.model_id,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=ratio, image_size=res),
                ),
            )

            if response.parts:
                for part in response.parts:
                    if part.inline_data:
                        part.as_image().save(image_path)

                        # Sanity check: make sure we didn't write an empty/corrupt file
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                            return True
                        logger.warning("Saved file is empty, treating as failure: %s", image_path)

        except Exception as e:
            err = str(e)

            if "503" in err:
                logger.warning("503 (service unavailable) on attempt %d — skipping row", attempt)
                return False

            if "429" in err:
                wait = 10 * attempt  # exponential-ish backoff: 10s, 20s, 30s...
                logger.warning("429 (rate limited) — waiting %ds before retry", wait)
                time.sleep(wait)
                continue

            logger.warning("Attempt %d/%d failed: %s", attempt, cfg.max_attempts, e)
            time.sleep(2)

    return False


# =========================================================
# MAIN
# =========================================================
def run_automation(cfg: Config) -> None:
    logger = setup_logging(os.path.join(".", cfg.log_file))

    os.makedirs(cfg.output_dir, exist_ok=True)
    client = build_client()

    if not os.path.exists(cfg.csv_file):
        raise FileNotFoundError(f"Input CSV not found: {cfg.csv_file}")

    df = pd.read_csv(cfg.csv_file)

    required_cols = {"title", "caption"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.head(cfg.max_rows).copy()

    for col, default in (("generated_image_path", ""), ("quality_tier", "")):
        if col not in df.columns:
            df[col] = default
        # Force object dtype so later string assignments (df.at[idx, col] = "...")
        # never hit pandas' "column is float64 because it was all NaN" trap.
        df[col] = df[col].astype(object)

    # ---- Resume support: reload existing output CSV if present ----
    if os.path.exists(cfg.output_csv):
        prior = pd.read_csv(cfg.output_csv)
        if len(prior) == len(df):
            df["generated_image_path"] = prior["generated_image_path"].astype(object).where(
                prior["generated_image_path"].notna(), ""
            )
            df["quality_tier"] = prior["quality_tier"].astype(object).where(
                prior["quality_tier"].notna(), ""
            )
            logger.info("Resumed from existing output CSV (%s).", cfg.output_csv)

    logger.info("Processing %d rows...", len(df))

    n_success, n_fail, n_skipped = 0, 0, 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        filename = f"fake_news_{index}.png"
        image_path = os.path.join(cfg.output_dir, filename)

        # Skip rows already generated successfully (resume-safe)
        already_done = (
            str(row.get("generated_image_path", "")).strip() != ""
            and os.path.exists(str(row.get("generated_image_path", "")))
        )
        if already_done:
            n_skipped += 1
            continue

        fake_title = str(row.get("title", "")).strip()
        fake_caption = str(row.get("caption", "")).strip()

        if not fake_title:
            logger.warning("Row %d has empty fake_title — skipping", index)
            df.at[index, "quality_tier"] = "skipped_empty_title"
            continue

        prompt, ratio, res, tier = build_prompt(fake_title, fake_caption)

        success = generate_one_image(client, cfg, prompt, ratio, res, image_path, logger)

        if success:
            df.at[index, "generated_image_path"] = image_path
            df.at[index, "quality_tier"] = tier
            n_success += 1
            logger.info("[%d] Saved (%s)", index + 1, tier)
        else:
            df.at[index, "generated_image_path"] = ""
            df.at[index, "quality_tier"] = "failed"
            n_fail += 1
            logger.info("[%d] Failed", index + 1)

        # Checkpoint after every row so a crash never loses progress
        df.to_csv(cfg.output_csv, index=False)

        time.sleep(cfg.base_sleep)

    logger.info(
        "DONE — success=%d, failed=%d, skipped(already done)=%d, total=%d",
        n_success, n_fail, n_skipped, len(df),
    )


# =========================================================
# CLI
# =========================================================
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Generate fake-news images from a CSV dataset.")
    p.add_argument("--csv", default="fake_news_merged.csv", help="Input CSV path")
    p.add_argument("--output-dir", default="generated_images", help="Output directory")
    p.add_argument("--max-rows", type=int, default=4432, help="Max rows to process")
    p.add_argument("--sleep", type=float, default=1.5, help="Seconds between requests")
    p.add_argument("--attempts", type=int, default=3, help="Retries per row")
    p.add_argument("--model", default="gemini-3.1-flash-image-preview", help="Model ID")
    args = p.parse_args()

    return Config(
        csv_file=args.csv,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        base_sleep=args.sleep,
        max_attempts=args.attempts,
        model_id=args.model,
    )


if __name__ == "__main__":
    run_automation(parse_args())