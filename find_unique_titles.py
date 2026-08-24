import pandas as pd

try:
    df = pd.read_csv('real_news.csv')
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print("Error: 'real_news.csv' not found. Please make sure the file is in the correct directory.")
    df = None

if df is not None:
    if 'title' in df.columns:
        unique_titles = df['title'].unique()
        print(f"Found {len(df['title'])} titles.")
        print(f"Found {len(unique_titles)} unique titles.")
    else:
        print("Error: 'title' column not found in the DataFrame. Available columns are:")
        print(df.columns)
