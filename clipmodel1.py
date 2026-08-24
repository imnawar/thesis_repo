"""
CLIP-based semantic/cross-modal consistency branch (Branch 1) — ARABIC VERSION, v2

Changes from your uploaded version:
  - FIXED: NewsPairDataset and evaluate_by_tier referenced `.text`, which no
    longer exists after you renamed the column to `title` — both now use
    `.title` consistently, matching load_data() and zero_shot_baseline().
  - ADDED: a stratified train/val/test split (split_data), since a
    supervised classifier trained and evaluated on the same rows gives
    meaningless numbers.
  - ADDED: Branch1Classifier — a small MLP head on top of (image_feat,
    text_feat, |image_feat - text_feat|) that outputs an actual real/fake
    logit, trained with BCEWithLogitsLoss against your `label` column.
  - ADDED: train_classifier() / evaluate_classifier() — the supervised
    training loop and accuracy/F1/AUC evaluation for the standalone
    Branch-1 ablation row in your results table.

Everything from your uploaded version (mps device, local M-CLIP snapshot
path, manual encode_text implementation, Arabic normalization) is kept as-is.

Pipeline stages:
  1. load_data()                -> CSV with image_path, title (Arabic), label, [topic]
  2. split_data()                -> stratified train/val/test
  3. zero_shot_baseline()        -> pretrained M-CLIP + CLIP similarity, no training
  4. NewsPairDataset + train()   -> contrastive fine-tuning of text projection head (unsupervised-style, tiered hard negatives)
  5. Branch1Classifier + train_classifier() -> SUPERVISED classification head on top of embeddings
  6. evaluate_classifier()       -> accuracy / F1 / AUC on held-out test set
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import open_clip
from multilingual_clip import pt_multilingual_clip
import transformers

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
TEXT_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
IMAGE_MODEL_NAME = "ViT-B-32"
IMAGE_PRETRAINED = "openai"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-5
CLASSIFIER_LR = 1e-3
EPOCHS = 5
CLASSIFIER_EPOCHS = 15

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "merged_news.csv")

TEXT_MODEL_LOCAL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--M-CLIP--XLM-Roberta-Large-Vit-B-32/"
    "snapshots/85949a3ac1415027bdf7a5d0994bba8d992fe494"
)

# ---------------------------------------------------------------------------
# Arabic text normalization
# ---------------------------------------------------------------------------
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")

def normalize_arabic(text: str) -> str:
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
    df = df.dropna(subset=["image_path"]).copy()
    df = df[df["image_path"].astype(str).str.strip() != ""].copy()
    df["image_path"] = df["image_path"].astype(str).apply(
        lambda p: p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
    )
    # Drop rows with empty text after normalization too
    df = df[df["title"].str.strip() != ""].copy()
    return df.reset_index(drop=True)


def split_data(df, test_size=0.2, val_size=0.1, seed=42):
    """
    Stratified split so real/fake ratio is preserved in each split.
    test_size and val_size are both fractions of the FULL dataset.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_fraction_of_train_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction_of_train_val,
        stratify=train_val["label"], random_state=seed
    )
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading — text and image towers are SEPARATE objects here
# ---------------------------------------------------------------------------
def load_models(text_model_path=TEXT_MODEL_LOCAL_PATH,
                 text_model_name=TEXT_MODEL_NAME,
                 image_model_name=IMAGE_MODEL_NAME,
                 image_pretrained=IMAGE_PRETRAINED):
    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
        text_model_path, local_files_only=True
    )
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)
    text_model.to(DEVICE)

    image_model, _, image_preprocess = open_clip.create_model_and_transforms(
        image_model_name, pretrained=image_pretrained
    )
    image_model.to(DEVICE)

    return text_model, text_tokenizer, image_model, image_preprocess


def encode_text(text_model, text_tokenizer, texts):
    """Manual M-CLIP forward pass (tokenize -> transformer -> mean-pool -> projection)."""
    txt_tok = text_tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    txt_tok = {k: v.to(DEVICE) for k, v in txt_tok.items()}

    embs = text_model.transformer(**txt_tok)[0]
    attention_mask = txt_tok["attention_mask"].unsqueeze(-1)
    embs = (embs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    embs = text_model.LinearTransformation(embs)
    return embs


def encode_image(image_model, image_tensor):
    return image_model.encode_image(image_tensor)


# ---------------------------------------------------------------------------
# 2. Zero-shot baseline
# ---------------------------------------------------------------------------
def zero_shot_baseline(df, text_model, text_tokenizer, image_model, image_preprocess):
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

            sims.append((img_feat @ txt_feat.T).item())

    df = df.copy()
    df["clip_similarity"] = sims
    return df


def report_zero_shot_separation(df_with_sims):
    valid = df_with_sims.dropna(subset=["clip_similarity"])
    auc = roc_auc_score(valid["label"], -valid["clip_similarity"])
    print(f"Zero-shot M-CLIP similarity AUC (fake detection): {auc:.4f}")
    return auc


# ---------------------------------------------------------------------------
# 3. Contrastive fine-tuning (tiered hard negatives) — optional, unsupervised-style
# ---------------------------------------------------------------------------
class NewsPairDataset(Dataset):
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
        anchor_text = str(anchor.title)   # FIXED: was anchor.text

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

        return anchor_img, neg_img, anchor_text


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


def train_contrastive(df, text_model, text_tokenizer, image_model, image_preprocess,
                       epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    for param in image_model.parameters():
        param.requires_grad = False
    for name, param in text_model.named_parameters():
        param.requires_grad = "LinearTransformation" in name

    trainable_params = [p for p in text_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No trainable parameters found — inspect text_model.named_parameters() "
            "and adjust the freeze/unfreeze filter to match your installed version."
        )

    temperature = nn.Parameter(torch.tensor(0.07, device=DEVICE))
    optimizer = torch.optim.AdamW(trainable_params + [temperature], lr=lr)

    dataset = NewsPairDataset(df, image_preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, collate_fn=collate_fn)

    image_model.eval()
    text_model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for anchor_img, neg_img, texts in tqdm(loader, desc=f"Contrastive epoch {epoch+1}/{epochs}"):
            anchor_img = anchor_img.to(DEVICE)
            with torch.no_grad():
                img_feat = encode_image(image_model, anchor_img)
            txt_feat = encode_text(text_model, text_tokenizer, texts)

            loss = contrastive_loss(img_feat, txt_feat, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: avg contrastive loss = {total_loss / len(loader):.4f}")

    return text_model


# ---------------------------------------------------------------------------
# 4. Branch-1 standalone classifier (OPTION A) — supervised, produces a
#    real/fake decision directly from this branch alone.
# ---------------------------------------------------------------------------
class Branch1Classifier(nn.Module):
    """
    MLP head on top of concatenated (image_feat, text_feat, |image_feat - text_feat|).
    The absolute-difference term explicitly gives the classifier an easy
    handle on cross-modal disagreement, rather than making it rediscover
    that concept from the raw concatenation alone.
    """
    def __init__(self, embed_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img_feat, txt_feat):
        diff = torch.abs(img_feat - txt_feat)
        combined = torch.cat([img_feat, txt_feat, diff], dim=-1)
        return self.net(combined).squeeze(-1)  # raw logit


class EmbeddingCache:
    """
    Precomputes and caches CLIP embeddings for a split so the classifier
    training loop doesn't re-run the (slow) CLIP forward pass every epoch.
    Encoders are frozen for this stage, so embeddings are constant.
    """
    def __init__(self, df, text_model, text_tokenizer, image_model, image_preprocess,
                 batch_size=32):
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)
        img_feats, txt_feats = [], []

        text_model.eval()
        image_model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(df), batch_size), desc="Caching embeddings"):
                chunk = df.iloc[start:start + batch_size]
                images = torch.stack([
                    image_preprocess(Image.open(p).convert("RGB")) for p in chunk.image_path
                ]).to(DEVICE)
                texts = [str(t) for t in chunk.title]

                img_feat = F.normalize(encode_image(image_model, images), dim=-1)
                txt_feat = F.normalize(encode_text(text_model, text_tokenizer, texts), dim=-1)

                img_feats.append(img_feat.cpu())
                txt_feats.append(txt_feat.cpu())

        self.img_feats = torch.cat(img_feats, dim=0)
        self.txt_feats = torch.cat(txt_feats, dim=0)

    def __len__(self):
        return len(self.labels)


def train_classifier(train_cache, val_cache, embed_dim,
                      epochs=CLASSIFIER_EPOCHS, lr=CLASSIFIER_LR, batch_size=64):
    classifier = Branch1Classifier(embed_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    n_train = len(train_cache)
    best_val_auc = -1
    best_state = None

    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            img_feat = train_cache.img_feats[idx].to(DEVICE)
            txt_feat = train_cache.txt_feats[idx].to(DEVICE)
            labels = train_cache.labels[idx].to(DEVICE)

            logits = classifier(img_feat, txt_feat)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        # Validation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_cache.img_feats.to(DEVICE), val_cache.txt_feats.to(DEVICE))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(val_cache.labels.numpy(), val_probs)

        print(f"Epoch {epoch+1}: train loss = {total_loss/n_train:.4f}, val AUC = {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

    classifier.load_state_dict(best_state)
    print(f"Best val AUC: {best_val_auc:.4f} — restored best checkpoint.")
    return classifier


def evaluate_classifier(classifier, test_cache, threshold=0.5):
    """Reports the standard metrics for your Branch-1 ablation row."""
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_cache.img_feats.to(DEVICE), test_cache.txt_feats.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        labels = test_cache.labels.numpy()

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
    }
    print("Branch-1 standalone classifier — test set results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    text_model, text_tokenizer, image_model, image_preprocess = load_models()

    print("=== Step 1: Zero-shot M-CLIP baseline (on full data, for reference) ===")
    df_scored = zero_shot_baseline(df, text_model, text_tokenizer, image_model, image_preprocess)
    report_zero_shot_separation(df_scored)

    # Optional: contrastive fine-tuning of the text projection head before
    # caching embeddings for the classifier. Uncomment to include it —
    # report both variants (with/without contrastive pretraining) as an
    # ablation if you run this.
    # text_model = train_contrastive(train_df, text_model, text_tokenizer, image_model, image_preprocess)

    print("\n=== Step 2: Cache embeddings for train/val/test ===")
    train_cache = EmbeddingCache(train_df, text_model, text_tokenizer, image_model, image_preprocess)
    val_cache = EmbeddingCache(val_df, text_model, text_tokenizer, image_model, image_preprocess)
    test_cache = EmbeddingCache(test_df, text_model, text_tokenizer, image_model, image_preprocess)

    embed_dim = train_cache.img_feats.shape[1]

    print("\n=== Step 3: Train Branch-1 standalone classifier ===")
    classifier = train_classifier(train_cache, val_cache, embed_dim)

    print("\n=== Step 4: Evaluate on held-out test set ===")
    evaluate_classifier(classifier, test_cache)

# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# Why embeddings are cached rather than recomputed each epoch: the image/text
# encoders are frozen for classifier training (only the small MLP head
# trains), so their outputs are constant across epochs. Caching turns a slow
# CLIP forward pass into a fast lookup and makes 15+ epochs of classifier
# training practical on CPU/MPS instead of requiring a CUDA GPU.
#
# If you DO run the contrastive fine-tuning step first (train_contrastive),
# re-cache embeddings afterward using the updated text_model — the cache
# above assumes frozen encoders and will be stale otherwise.
#
# This classifier is your Branch-1-ONLY ablation result. Your main result
# will come from a fusion classifier that also takes Branch 2 (image
# artifacts) and Branch 3 (text artifacts) features as additional inputs —
# structurally, that's the same Branch1Classifier pattern, just with a wider
# input layer taking all three branches' features concatenated together.