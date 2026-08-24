import os
import re
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


def _sort_key(filename):
    """
    Sort actual image files so that:
      - 'main_image.*' always comes first
      - numeric filenames (1.jpg, 2.jpg, 3.jpg, ...) come after, in numeric order
      - anything else falls back to alphabetical, after the numeric ones
    """
    name, _ext = os.path.splitext(filename)
    if name.lower() == "main_image":
        return (0, 0)
    match = re.fullmatch(r"\d+", name)
    if match:
        return (1, int(name))
    return (2, name)


def get_actual_images(images_folder):
    """Return the real list of image filenames present in the images folder, correctly ordered."""
    if not os.path.isdir(images_folder):
        return []
    valid_ext = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    files = [
        f for f in os.listdir(images_folder)
        if f.lower().endswith(valid_ext) and os.path.isfile(os.path.join(images_folder, f))
    ]
    files.sort(key=_sort_key)
    return files


def build_dataset(root_folder):
    data = []
    # Loop over each news folder ONLY
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

        # Parse text file (title, captions, urls, date, article text)
        title, _images_from_txt, captions, image_urls, date, full_text = parse_context_file(context_path)

        if not title:
            print(f"Warning: No title in {folder_name}")

        # 🔧 FIX: use the actual files that exist on disk instead of trusting
        # the filenames written inside content.txt, since real folders don't
        # always follow the 1.jpg/2.jpg/3.jpg convention (some use
        # main_image.jpg + 2.jpg + 3.jpg, etc.)
        actual_images = get_actual_images(images_folder)

        if not actual_images:
            print(f"Skipping (no image files found on disk): {folder_name}")
            continue

        # Match items safely against whichever list is shortest
        num_items = min(len(actual_images), len(captions), len(image_urls))
        if num_items == 0:
            print(f"Skipping (no valid pairs): {folder_name}")
            continue

        # If there are more actual images than captions/urls, note it (optional debug)
        if len(actual_images) != num_items:
            print(f"Note: image/caption/url count mismatch in {folder_name} "
                  f"(images={len(actual_images)}, captions={len(captions)}, urls={len(image_urls)})")

        for i in range(num_items):
            img_path = os.path.join(images_folder, actual_images[i])
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