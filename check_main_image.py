import os

def check_main_image(root_folder):
    missing = []
    present = []
    no_images_folder = []

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        images_folder = os.path.join(folder_path, "images")

        if not os.path.isdir(images_folder):
            no_images_folder.append(folder_name)
            continue

        # Case-insensitive check for main_image.<ext>
        files = os.listdir(images_folder)
        has_main = any(
            os.path.splitext(f)[0].lower() == "main_image"
            for f in files
        )

        if has_main:
            present.append(folder_name)
        else:
            missing.append(folder_name)

    return present, missing, no_images_folder


if __name__ == "__main__":
    root = "aljazeera"  # make sure this path is correct

    present, missing, no_images_folder = check_main_image(root)

    total = len(present) + len(missing) + len(no_images_folder)

    print(f"Total folders checked: {total}")
    print(f"Folders WITH main_image.*: {len(present)}")
    print(f"Folders MISSING main_image.*: {len(missing)}")
    print(f"Folders with NO images/ subfolder at all: {len(no_images_folder)}")

    if missing:
        print("\n--- Folders missing main_image.* ---")
        for f in missing:
            print(f)

    if no_images_folder:
        print("\n--- Folders with no images/ subfolder ---")
        for f in no_images_folder:
            print(f)

    # Optional: save the lists to text files for easier review
    with open("folders_missing_main_image.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(missing))

    with open("folders_no_images_folder.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(no_images_folder))