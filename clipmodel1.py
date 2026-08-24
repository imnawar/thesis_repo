"""
CLIP-based semantic/cross-modal consistency branch (Branch 1) — ARABIC VERSION

Key difference from the English version: plain OpenAI CLIP's text tower was
trained almost entirely on English and gives near-meaningless similarity
scores on Arabic. This version uses M-CLIP (an XLM-RoBERTa text encoder
distilled to align with CLIP's *existing* image embedding space), paired
with the original OpenAI CLIP image encoder — the two are separate model
objects here, not one joint model, so they're loaded and called differently
than the English pipeline.

Pipeline stages (same structure as before):
  1. load_data()               -> CSV with image_path, text (Arabic), label, [topic], [subtype]
  2. normalize_arabic()        -> lightweight text normalization before encoding
  3. zero_shot_baseline()      -> pretrained M-CLIP + CLIP similarity, no training
  4. NewsPairDataset + train() -> fine-tunes the text projection head with tiered hard negatives
  5. evaluate_by_tier()        -> AUC/accuracy broken out by negative-tier difficulty
"""

import re
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import open_clip
from multilingual_clip import pt_multilingual_clip
import transformers

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
# M-CLIP text model — must match the image encoder below. Options (largest/best
# first): 'M-CLIP/XLM-Roberta-Large-Vit-B-32', 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
TEXT_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
IMAGE_MODEL_NAME = "ViT-B-32"     # must match the Vit-B-32 / Vit-L-14 in TEXT_MODEL_NAME
IMAGE_PRETRAINED = "openai"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 5
# CSV_PATH = "merged_real_fake_news_cleaned.csv"     # columns: image_path, text (Arabic), label, topic (optional)
# DATASET_ROOT = "/Users/manalnawar/Desktop/conda_fake_news_env1"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "merged_news.csv")

# ---------------------------------------------------------------------------
# Arabic text normalization
# ---------------------------------------------------------------------------
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # tashkeel + tatweel

def normalize_arabic(text: str) -> str:
    """
    Light, defensible normalization — NOT dialect conversion. Just removes
    noise that inflates surface-level variance without changing meaning:
      - diacritics (tashkeel) and tatweel (kashida elongation)
      - alef variants (إ أ آ -> ا)
      - alef maqsura (ى -> ي), ta marbuta left as-is (changes meaning if stripped)
      - repeated whitespace, stray punctuation duplication
    Report this step explicitly in your methodology — it's a modeling choice,
    not a neutral default.
    """
    if not isinstance(text, str):
        return ""
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    assert {"image_path", "title", "label"}.issubset(df.columns), \
        "CSV must have at least: image_path, title, label"

    df["title"] = df["title"].apply(normalize_arabic)

    # Convert relative image paths to absolute paths
    # df["image_path"] = df["image_path"].astype(str).str.strip()

    # df["image_path"] = df["image_path"].apply(
    #     lambda p: p if os.path.isabs(p)
    #     else os.path.join(PROJECT_ROOT, p)
    # )
    # Remove rows without an image path
    df = df.dropna(subset=["image_path"]).copy()

    # Remove empty/whitespace-only image paths
    df = df[df["image_path"].astype(str).str.strip() != ""].copy()

    # Convert relative image paths to absolute paths
    df["image_path"] = df["image_path"].astype(str).apply(
        lambda p: p if os.path.isabs(p)
        else os.path.join(PROJECT_ROOT, p)
    )

    return df


# ---------------------------------------------------------------------------
# Model loading — text and image towers are SEPARATE objects here
# ---------------------------------------------------------------------------
def load_models(text_model_name=TEXT_MODEL_NAME,
                 image_model_name=IMAGE_MODEL_NAME,
                 image_pretrained=IMAGE_PRETRAINED):
    text_model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--M-CLIP--XLM-Roberta-Large-Vit-B-32/"
        "snapshots/85949a3ac1415027bdf7a5d0994bba8d992fe494"
    )

    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
        text_model_path,
        local_files_only=True
    )
    # text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(text_model_name)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)
    text_model.to(DEVICE)

    image_model, _, image_preprocess = open_clip.create_model_and_transforms(
        image_model_name, pretrained=image_pretrained
    )
    image_model.to(DEVICE)

    return text_model, text_tokenizer, image_model, image_preprocess


def encode_text(text_model, text_tokenizer, texts):
    """
    Encode Arabic text using M-CLIP while enforcing XLM-R's
    maximum sequence length of 512 tokens.
    """

    # Tokenize with truncation
    txt_tok = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    txt_tok = {k: v.to(DEVICE) for k, v in txt_tok.items()}

    # Same computation used by multilingual-clip's forward()
    embs = text_model.transformer(**txt_tok)[0]

    # Mean pooling using attention mask
    attention_mask = txt_tok["attention_mask"].unsqueeze(-1)
    embs = (embs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    # M-CLIP projection: 1024 -> 512
    embs = text_model.LinearTransformation(embs)

    return embs


def encode_image(image_model, image_tensor):
    return image_model.encode_image(image_tensor)


# ---------------------------------------------------------------------------
# 2. Zero-shot baseline — run this FIRST
# ---------------------------------------------------------------------------
def zero_shot_baseline(df):
    text_model, text_tokenizer, image_model, image_preprocess = load_models()
    text_model.eval()
    image_model.eval()

    sims = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot M-CLIP"):
            try:
                image = image_preprocess(Image.open(row.image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            except Exception as e:
                print(f"Skipping {row.image_path}: {e}")
                sims.append(np.nan)
                continue

            img_feat = encode_image(image_model, image)
            txt_feat = encode_text(text_model, text_tokenizer, [str(row.title)])

            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)

            sim = (img_feat @ txt_feat.T).item()
            sims.append(sim)

    df = df.copy()
    df["clip_similarity"] = sims
    return df


def report_zero_shot_separation(df_with_sims):
    valid = df_with_sims.dropna(subset=["clip_similarity"])
    auc = roc_auc_score(valid["label"], -valid["clip_similarity"])
    print(f"Zero-shot M-CLIP similarity AUC (fake detection): {auc:.4f}")
    print("A low AUC here is expected and motivates fine-tuning — M-CLIP is a")
    print("general-purpose multilingual alignment model, not tuned for Arabic")
    print("news specifically or for entity-level consistency checks.")
    return auc


# ---------------------------------------------------------------------------
# 3. Fine-tuning — only the text projection head is trained by default.
#    The image tower and M-CLIP's underlying XLM-R backbone stay frozen;
#    training the full text backbone on a thesis-scale dataset risks
#    catastrophic forgetting of the alignment M-CLIP was distilled for.
# ---------------------------------------------------------------------------
class NewsPairDataset(Dataset):
    """Same tiered hard-negative logic as the English version."""
    def __init__(self, df, image_preprocess, tier_probs=(0.5, 0.3, 0.2)):
        self.real_df = df[df.label == 0].reset_index(drop=True)
        self.fake_df = df[df.label == 1].reset_index(drop=True)
        self.image_preprocess = image_preprocess
        self.tier_probs = tier_probs

        self.has_topic = "topic" in df.columns
        if self.has_topic:
            self.topic_index = {
                t: g.index.tolist() for t, g in self.real_df.groupby("topic")
            }

    def __len__(self):
        return len(self.real_df)

    def _load_image(self, path):
        return self.image_preprocess(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        anchor = self.real_df.iloc[idx]
        anchor_img = self._load_image(anchor.image_path)
        anchor_text = str(anchor.text)

        r = random.random()
        t1, t2, _ = self.tier_probs

        if r < t1 or not self.has_topic:
            neg_idx = random.randrange(len(self.real_df))
            neg_img = self._load_image(self.real_df.iloc[neg_idx].image_path)
        elif r < t1 + t2:
            candidates = self.topic_index.get(anchor.topic, [idx])
            candidates = [c for c in candidates if c != idx] or [idx]
            neg_idx = random.choice(candidates)
            neg_img = self._load_image(self.real_df.iloc[neg_idx].image_path)
        else:
            if len(self.fake_df) > 0:
                neg_row = self.fake_df.iloc[random.randrange(len(self.fake_df))]
                neg_img = self._load_image(neg_row.image_path)
            else:
                neg_idx = random.randrange(len(self.real_df))
                neg_img = self._load_image(self.real_df.iloc[neg_idx].image_path)

        return anchor_img, neg_img, anchor_text  # text stays a string; tokenized in collate


def collate_fn(batch):
    anchor_imgs, neg_imgs, texts = zip(*batch)
    return torch.stack(anchor_imgs), torch.stack(neg_imgs), list(texts)


def contrastive_loss(img_feat, txt_feat, temperature):
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    logits = img_feat @ txt_feat.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def train(df, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    text_model, text_tokenizer, image_model, image_preprocess = load_models()

    # Freeze everything except the M-CLIP text model's linear projection head
    # (named 'LinearTransformation' in the multilingual-clip implementation).
    for param in image_model.parameters():
        param.requires_grad = False
    for name, param in text_model.named_parameters():
        param.requires_grad = "LinearTransformation" in name

    trainable_params = [p for p in text_model.parameters() if p.requires_grad]
    if not trainable_params:
        # Fallback: check the actual attribute name in your installed version
        # (pip show multilingual-clip / inspect text_model.named_parameters())
        # and update the filter above accordingly before training.
        raise RuntimeError(
            "No trainable parameters found — inspect text_model.named_parameters() "
            "and adjust the freeze/unfreeze filter above to match your installed version."
        )

    temperature = nn.Parameter(torch.tensor(0.07, device=DEVICE))
    optimizer = torch.optim.AdamW(trainable_params + [temperature], lr=lr)

    dataset = NewsPairDataset(df, image_preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, collate_fn=collate_fn)

    image_model.eval()
    text_model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for anchor_img, neg_img, texts in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchor_img = anchor_img.to(DEVICE)

            with torch.no_grad():
                img_feat = encode_image(image_model, anchor_img)

            txt_feat = encode_text(text_model, text_tokenizer, texts)

            loss = contrastive_loss(img_feat, txt_feat, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: avg loss = {total_loss / len(loader):.4f}")

    return text_model, text_tokenizer, image_model, image_preprocess


# ---------------------------------------------------------------------------
# 4. Evaluate by tier/subtype
# ---------------------------------------------------------------------------
def evaluate_by_tier(text_model, text_tokenizer, image_model, image_preprocess, test_df):
    text_model.eval()
    image_model.eval()
    results = {}
    for tier_name, group in test_df.groupby("subtype"):
        sims, labels = [], []
        with torch.no_grad():
            for _, row in group.iterrows():
                image = image_preprocess(Image.open(row.image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
                img_feat = F.normalize(encode_image(image_model, image), dim=-1)
                txt_feat = F.normalize(encode_text(text_model, text_tokenizer, [str(row.text)]), dim=-1)
                sims.append((img_feat @ txt_feat.T).item())
                labels.append(row.label)
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, [-s for s in sims])
            results[tier_name] = auc
            print(f"{tier_name}: AUC = {auc:.4f}  (n={len(group)})")
        else:
            print(f"{tier_name}: skipped (only one class present, n={len(group)})")
    return results


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    print("=== Step 1: Zero-shot M-CLIP baseline ===")
    df_scored = zero_shot_baseline(df)
    report_zero_shot_separation(df_scored)

    # print("\n=== Step 2: Fine-tune text projection head with tiered hard negatives ===")
    # text_model, text_tokenizer, image_model, image_preprocess = train(df)

    # print("\n=== Step 3: Evaluate by subtype/tier ===")
    # evaluate_by_tier(text_model, text_tokenizer, image_model, image_preprocess, df)

# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# Dialect vs. MSA: M-CLIP's XLM-R backbone was trained on general multilingual
# web text, which skews MSA-heavy. If your dataset includes dialectal Arabic
# (Egyptian, Gulf, Levantine, Maghrebi), expect weaker zero-shot performance
# on dialectal examples specifically — worth reporting as a breakdown
# (MSA vs. dialect subsets) alongside the tier breakdown, same rationale as
# the AraNews MSA-only limitation discussed earlier in this thesis.
#
# Alternative worth benchmarking: AraCLIP / Arabic-adapted CLIP checkpoints
# (search Hugging Face for recent Arabic-native CLIP models — this space
# moves quickly, check for options newer than what's listed here) trained
# directly on Arabic image-text pairs rather than distilled from English
# CLIP. Running both M-CLIP and an Arabic-native CLIP as parallel baselines
# is a strong, cheap addition to your results table.
#
# Compute note: same as the English version — projection-head-only fine-
# tuning is feasible on a single consumer GPU. Unfreezing more of the XLM-R
# backbone requires substantially more memory and data to avoid overfitting.