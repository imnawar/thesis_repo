# AraFakeNews human-evaluation Streamlit app
# Extracted from the final %%writefile app.py cell in Untitled135.ipynb.
# Deployment adaptation: data/image paths are resolved relative to this repository
# so the app is not dependent on Google Colab's /content/... paths.

import streamlit as st
import pandas as pd
import os
import random
import uuid
from datetime import datetime

st.set_page_config(page_title="تقييم واقعية الصور", layout="centered")

st.title("📰 تقييم مدى واقعية صور الأخبار المزيفة")
st.caption("ملاحظة: جميع الأخبار المعروضة في هذا الاستبيان **مُولّدة اصطناعياً (مزيفة)**. المطلوب هو تقييم مدى واقعية الصورة، وليس تحديد ما إذا كانت حقيقية أم لا.")

# ==========================================
# 🔹 CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "GeneratedImagesToEvaluate")
CSV_PATH = os.path.join(IMAGES_DIR, "generated_image_dataset_filtered.csv")
DEMOGRAPHICS_CSV_PATH = os.path.join(IMAGES_DIR, "user_demographics.csv")

MAX_IMAGES_PER_USER = 25

# ==========================================
# 🔹 Auto-generated user ID
# ==========================================
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"

user_id = st.session_state.user_id

st.markdown(f"👤 المستخدم الحالي: `{user_id}`")

# ==========================================
# 🔹 Demographics form (once per user, before evaluation starts)
# ==========================================
if "demographics_done" not in st.session_state:
    st.session_state.demographics_done = False

if not st.session_state.demographics_done:

    st.markdown("### 👋 قبل البدء، أخبرنا قليلاً عن نفسك")
    st.caption("هذه المعلومات تساعدنا في تحليل النتائج، ولن تُستخدم إلا لأغراض البحث العلمي.")

    with st.form("demographics_form"):

        age_group = st.selectbox(
            "الفئة العمرية",
            [
                "أقل من 18",
                "18 - 24",
                "25 - 34",
                "35 - 44",
                "45 - 54",
                "55 فأكثر",
            ]
        )

        gender = st.radio(
            "الجنس",
            ["ذكر", "أنثى"],
            horizontal=True
        )

        education_stage = st.selectbox(
            "المرحلة الدراسية",
            [
                "أقل من ثانوي",
                "ثانوي",
                "بكالوريوس / جامعي",
                "ماجستير",
                "دكتوراه",
                "أخرى",
            ]
        )

        field_of_study = st.selectbox(
            "المجال العلمي / التخصص",
            [
                "علمي / تقني (هندسة، حاسب، علوم، طب...)",
                "إنساني / اجتماعي (أدب، إعلام، تربية، شريعة...)",
                "إداري / اقتصادي",
                "أخرى",
                "لا ينطبق",
            ]
        )

        social_media_usage = st.selectbox(
            "ما مدى استخدامك لوسائل التواصل الاجتماعي؟",
            [
                "بشكل يومي ومتكرر",
                "يومياً",
                "عدة مرات أسبوعياً",
                "نادراً",
                "لا أستخدمها",
            ]
        )

        media_trust = st.select_slider(
            "بشكل عام، ما مدى ثقتك بوسائل الإعلام ومصادر الأخبار؟",
            options=["منخفضة جداً", "منخفضة", "متوسطة", "عالية", "عالية جداً"],
            value="متوسطة"
        )

        news_verification_habit = st.radio(
            "هل تتحقق عادةً من صحة الأخبار قبل تصديقها أو مشاركتها؟",
            ["دائماً", "أحياناً", "نادراً", "أبداً"],
            horizontal=True
        )

        submitted = st.form_submit_button("ابدأ التقييم")

        if submitted:

            demo_row = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "age_group": age_group,
                "gender": gender,
                "education_stage": education_stage,
                "field_of_study": field_of_study,
                "social_media_usage": social_media_usage,
                "media_trust": media_trust,
                "news_verification_habit": news_verification_habit,
            }

            # Append to demographics CSV (create if not exists)
            if os.path.exists(DEMOGRAPHICS_CSV_PATH):
                demo_df = pd.read_csv(DEMOGRAPHICS_CSV_PATH)
                demo_df = pd.concat(
                    [demo_df, pd.DataFrame([demo_row])],
                    ignore_index=True
                )
            else:
                demo_df = pd.DataFrame([demo_row])

            demo_df.to_csv(DEMOGRAPHICS_CSV_PATH, index=False)

            st.session_state.demographics_done = True
            st.rerun()

    st.stop()

# ==========================================
# 🔹 Load dataset
# ==========================================
df = pd.read_csv(CSV_PATH)

# Keep only rows with images
df = df[df["generated_image_path"].notna()]

# ==========================================
# 🔹 Create evaluation columns if missing
# ==========================================
required_cols = [
    "eval_image_1", "eval_image_2", "eval_image_3",
    "eval_title_1", "eval_title_2", "eval_title_3",
    "user_1", "user_2", "user_3",
    "eval_count",
    "final_score_image",
    "final_score_title"
]

for col in required_cols:

    if col not in df.columns:

        if col == "eval_count":
            df[col] = 0
        else:
            df[col] = ""

# Ensure correct dtypes regardless of what was inferred when reading the CSV
# (empty cells get read back as NaN/float64, which then rejects string writes)
text_cols = [
    "eval_image_1", "eval_image_2", "eval_image_3",
    "eval_title_1", "eval_title_2", "eval_title_3",
    "user_1", "user_2", "user_3",
    "final_score_image", "final_score_title"
]

for col in text_cols:
    df[col] = df[col].astype(object).where(df[col].notna(), "")

df["eval_count"] = pd.to_numeric(df["eval_count"], errors="coerce").fillna(0).astype(int)

# Save back so the columns persist even before the first rating is submitted
df.to_csv(CSV_PATH, index=False)

# ==========================================
# 🔹 Keep rows with < 3 evaluations
# ==========================================
remaining = df[df["eval_count"] < 3]

# ==========================================
# 🔹 Initialize random 25 images for user
# ==========================================
if "assigned_indices" not in st.session_state:

    available_indices = remaining.index.tolist()

    sample_size = min(MAX_IMAGES_PER_USER, len(available_indices))

    st.session_state.assigned_indices = random.sample(
        available_indices,
        sample_size
    )

    st.session_state.current_position = 0

# ==========================================
# 🔹 User completed all assigned images
# ==========================================
if st.session_state.current_position >= len(st.session_state.assigned_indices):

    st.balloons()
    st.success("🎉 انتهى التقييم، شكراً لمشاركتك!")
    st.stop()

# ==========================================
# 🔹 Current image
# ==========================================
current_idx = st.session_state.assigned_indices[
    st.session_state.current_position
]

current_row = df.loc[current_idx]

# ==========================================
# 🔹 Skip if already evaluated 3 times
# ==========================================
if int(current_row["eval_count"]) >= 3:

    st.session_state.current_position += 1
    st.rerun()

# ==========================================
# 🔹 Progress bar
# ==========================================
progress = (
    st.session_state.current_position + 1
) / len(st.session_state.assigned_indices)

st.progress(progress)

st.markdown(
    f"### 🧾 الخبر رقم "
    f"{st.session_state.current_position + 1}"
    f" / {len(st.session_state.assigned_indices)}"
)

# ==========================================
# 🔹 Display title
# ==========================================
st.subheader(current_row["title"])

# ==========================================
# 🔹 Display image
# ==========================================
img_path = current_row["generated_image_path"]

# Support both the old Colab /content/... paths and relative filenames.
if pd.notna(img_path):
    img_path = str(img_path)
    if not os.path.isabs(img_path) or not os.path.exists(img_path):
        img_path = os.path.join(IMAGES_DIR, os.path.basename(img_path))

if pd.notna(img_path) and os.path.exists(img_path):
    st.image(img_path)
else:
    st.warning("⚠️ الصورة غير موجودة")

# ==========================================
# 🔹 Display caption
# ==========================================
if (
    "caption" in current_row
    and pd.notna(current_row["caption"])
):
    st.markdown("### 📝 الوصف")
    st.write(current_row["caption"])

# ==========================================
# 🔹 Rating
# ==========================================
st.markdown("### إلى أي مدى تبدو هذه الصورة واقعية؟")
st.caption("تذكير: هذا الخبر والصورة المرفقة مُولَّدان اصطناعياً وليسا حقيقيَّين.")

image_rating = st.slider(
    "تقييم الصورة (1 = تبدو مزيفة بشكل واضح، 5 = تبدو واقعية جداً):",
    min_value=1,
    max_value=5,
    value=3
)

st.markdown("### إلى أي مدى يبدو هذا العنوان واقعياً؟")

title_rating = st.slider(
    "تقييم العنوان (1 = يبدو مزيفاً بشكل واضح، 5 = يبدو واقعياً جداً):",
    min_value=1,
    max_value=5,
    value=3
)

# ==========================================
# 🔹 Submit
# ==========================================
if st.button("إرسال"):

    eval_count = int(df.at[current_idx, "eval_count"])

    # --------------------------------------
    # Store evaluations separately
    # --------------------------------------
    if eval_count == 0:

        df.at[current_idx, "eval_image_1"] = image_rating
        df.at[current_idx, "eval_title_1"] = title_rating
        df.at[current_idx, "user_1"] = user_id

    elif eval_count == 1:

        df.at[current_idx, "eval_image_2"] = image_rating
        df.at[current_idx, "eval_title_2"] = title_rating
        df.at[current_idx, "user_2"] = user_id

    elif eval_count == 2:

        df.at[current_idx, "eval_image_3"] = image_rating
        df.at[current_idx, "eval_title_3"] = title_rating
        df.at[current_idx, "user_3"] = user_id

    # --------------------------------------
    # Increment evaluation count
    # --------------------------------------
    df.at[current_idx, "eval_count"] = eval_count + 1

    # --------------------------------------
    # Final average scores (image & title, separately)
    # --------------------------------------
    image_scores = [
        df.at[current_idx, "eval_image_1"],
        df.at[current_idx, "eval_image_2"],
        df.at[current_idx, "eval_image_3"]
    ]

    title_scores = [
        df.at[current_idx, "eval_title_1"],
        df.at[current_idx, "eval_title_2"],
        df.at[current_idx, "eval_title_3"]
    ]

    valid_image_scores = [s for s in image_scores if pd.notna(s) and s != ""]
    valid_title_scores = [s for s in title_scores if pd.notna(s) and s != ""]

    if len(valid_image_scores) == 3:
        avg_image_score = sum(map(float, valid_image_scores)) / 3
        df.at[current_idx, "final_score_image"] = round(avg_image_score, 2)

    if len(valid_title_scores) == 3:
        avg_title_score = sum(map(float, valid_title_scores)) / 3
        df.at[current_idx, "final_score_title"] = round(avg_title_score, 2)

    # --------------------------------------
    # Save updated CSV
    # --------------------------------------
    df.to_csv(CSV_PATH, index=False)

    st.success("✅ تم حفظ التقييم بنجاح")

    # --------------------------------------
    # Move to next image
    # --------------------------------------
    st.session_state.current_position += 1

    st.rerun()