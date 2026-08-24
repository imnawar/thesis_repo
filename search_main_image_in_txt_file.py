import os

def search_main_image_in_txts(root_folder, tag="[img]", target="main_image"):
    """
    Recursively search all .txt files under root_folder.
    Checks lines that start with the given tag (default '[img]')
    and flags files where the image name matches `target` (case-insensitive,
    any extension, e.g. main_image.jpg / main_image.png / MAIN_IMAGE.JPG).
    """
    matches = []      # (file_path, matching line)
    no_match_files = []
    total_txt_files = 0

    for dirpath, _dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if not fname.lower().endswith(".txt"):
                continue

            total_txt_files += 1
            file_path = os.path.join(dirpath, fname)

            found_in_file = False
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(tag):
                            img_name = line.replace(tag, "").strip()
                            # get just the base name without extension, case-insensitive
                            base_name = os.path.splitext(img_name)[0].lower()
                            if base_name == target.lower():
                                matches.append((file_path, line))
                                found_in_file = True
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            if not found_in_file:
                no_match_files.append(file_path)

    return matches, no_match_files, total_txt_files


if __name__ == "__main__":
    root = "aljazeera"  # change this to the folder containing your txt files

    matches, no_match_files, total_txt_files = search_main_image_in_txts(root)

    print(f"Total .txt files scanned: {total_txt_files}")
    print(f"Files containing a [img] line for main_image: {len(matches)}")
    print(f"Files WITHOUT a [img] line for main_image: {len(no_match_files)}")

    if matches:
        print("\n--- Files containing main_image (with the matching line) ---")
        for file_path, line in matches:
            print(f"{file_path}  ->  {line}")

    # Save results to text files for easier review
    with open("txt_files_with_main_image.txt", "w", encoding="utf-8") as f:
        for file_path, line in matches:
            f.write(f"{file_path}\t{line}\n")

    with open("txt_files_without_main_image.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(no_match_files))