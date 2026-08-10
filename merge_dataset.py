import pandas as pd

# --- Load the two CSV files ---
gemini_df = pd.read_csv("fake_news_1500_gemini.csv")
gpt_df = pd.read_csv("fake_news_1500_gpt-4_1.csv")
claude_df = pd.read_csv("fake_news_1500_claude.csv")

# --- Gemini file has no 'id' column, so create one ---
if "id" not in gemini_df.columns:
    gemini_df.insert(0, "id", range(1, len(gemini_df) + 1))

# --- Tag each row with its model source ---
gemini_df["model_source"] = "gemini"
gpt_df["model_source"] = "gpt-4.1"
claude_df["model_source"] = "claude"
# --- Merge (stack rows on top of each other) ---
merged_df = pd.concat([gpt_df, gemini_df, claude_df], ignore_index=True)

# --- Reorder columns for consistency ---
merged_df = merged_df[["id", "title", "caption", "topic", "country", "model_source"]]

# --- Save result ---
merged_df.to_csv("fake_news_merged.csv", index=False)

print(f"Merged shape: {merged_df.shape}")
print(merged_df["model_source"].value_counts())