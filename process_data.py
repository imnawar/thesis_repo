import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    # Basic sanity check
    print("Dataset loaded:", df.shape)
    print(df.columns)

    return df


import re

def normalize_arabic(text):
    if not isinstance(text, str):
        return ""

    # Remove diacritics
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # Normalize letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)

    return text.strip()


def clean_text(text):
    text = normalize_arabic(text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


from preprocessing import clean_text

def preprocess_dataframe(df):
    df["title_clean"] = df["title"].apply(clean_text)
    df["caption_clean"] = df["caption"].apply(clean_text)
    df["article_clean"] = df["article_text"].apply(clean_text)

    return df


from load_data import load_dataset
from preprocessing import clean_text

def preprocess_dataframe(df):
    df["title_clean"] = df["title"].apply(clean_text)
    df["caption_clean"] = df["caption"].apply(clean_text)
    return df


if __name__ == "__main__":
    df = load_dataset("real_news.csv")

    df = preprocess_dataframe(df)

    print(df[["title", "title_clean"]].head())
