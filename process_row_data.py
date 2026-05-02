import os
import pandas as pd

def parse_context_file(file_path):
    title = ""
    images = []
    captions = []
    image_urls = []
    article_text = []
    date = ""

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("[h1]"):
            title = line.replace("[h1]", "").strip()

        elif line.startswith("[published_date]"):
            date = line.replace("[published_date]", "").strip()

        elif line.startswith("[p]"):
            paragraph = line.replace("[p]", "").strip()
            if paragraph and paragraph not in ["حفظ", "شارِكْ"]:
                article_text.append(paragraph)

        elif line.startswith("[img]"):
            images.append(line.replace("[img]", "").strip())

        elif line.startswith("[img_src]"):
            image_urls.append(line.replace("[img_src]", "").strip())

        elif line.startswith("[caption]"):
            captions.append(line.replace("[caption]", "").strip())

    full_text = " ".join(article_text)

    return title, images, captions, image_urls, date, full_text


def build_dataset(root_folder):
    data = []

    # 🔥 Loop over each news folder ONLY
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Skip if not a folder
        if not os.path.isdir(folder_path):
            continue

        context_path = os.path.join(folder_path, "content.txt")
        images_folder = os.path.join(folder_path, "images")

        # Skip if context file doesn't exist
        if not os.path.exists(context_path):
            print(f"Skipping (no content): {folder_name}")
            continue

        # Parse file
        title, images, captions, image_urls, date, full_text = parse_context_file(context_path)

        # 🔴 Debug (important)
        if not title:
            print(f"Warning: No title in {folder_name}")

        # Match items safely
        num_items = min(len(images), len(captions), len(image_urls))

        if num_items == 0:
            print(f"Skipping (no valid pairs): {folder_name}")
            continue

        for i in range(num_items):
            img_path = os.path.join(images_folder, images[i])

            data.append({
                "id": folder_name,
                "title": title,
                "caption": captions[i],
                "image_path": img_path,
                "image_url": image_urls[i],
                "date": date,
                "article_text": full_text
            })

    return pd.DataFrame(data)


if __name__ == "__main__":
    root = "aljazeera"  # make sure this path is correct

    df = build_dataset(root)

    print("Total samples:", len(df))
    print(df.head())

    df.to_csv("real_news.csv", index=False, encoding="utf-8-sig")
