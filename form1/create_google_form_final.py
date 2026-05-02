import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# 🔹 SCOPES
SCOPES = ["https://www.googleapis.com/auth/forms.body"]

# 🔹 AUTH (LOCAL MACHINE ONLY)
flow = InstalledAppFlow.from_client_secrets_file(
    "credentials.json",
    SCOPES
)

# ✅ This opens browser automatically (works locally)
creds = flow.run_local_server(port=0)

forms_service = build("forms", "v1", credentials=creds)

# 🔹 LOAD DATA
df = pd.read_csv("evaluation_dataset_with_urls.csv")

# 🔹 (Recommended) Shuffle samples
df = df.sample(frac=1).reset_index(drop=True)

# 🔹 LIMIT samples (avoid long form)
LIMIT = 10
df = df.head(LIMIT)

# 🔹 CREATE FORM
form = {
    "info": {
        "title": "تقييم الأخبار (حقيقي أم مزيف)",
        "documentTitle": "تقييم الأخبار"
    }
}

form = forms_service.forms().create(body=form).execute()
form_id = form["formId"]

print("✅ Form created:")
print(f"https://docs.google.com/forms/d/{form_id}/edit")

# 🔹 BUILD FORM CONTENT
requests = []

for i, row in df.iterrows():

    # 🔹 Text (title + caption)
    requests.append({
        "createItem": {
            "item": {
                "title": f"العينة {i+1}",
                "description": f"العنوان:\n{row['fake_title']}\n\nالوصف:\n{row['fake_caption']}"
            },
            "location": {"index": i * 3}
        }
    })

    # 🔹 Image
    requests.append({
        "createItem": {
            "item": {
                "imageItem": {
                    "image": {
                        "sourceUri": row["image_url"]
                    }
                }
            },
            "location": {"index": i * 3 + 1}
        }
    })

    # 🔹 Question (Arabic scale)
    requests.append({
        "createItem": {
            "item": {
                "title": "هل هذا الخبر حقيقي أم مزيف؟ (5 = حقيقي، 1 = مزيف)",
                "questionItem": {
                    "question": {
                        "required": True,
                        "scaleQuestion": {
                            "low": 1,
                            "high": 5
                        }
                    }
                }
            },
            "location": {"index": i * 3 + 2}
        }
    })

# 🔹 APPLY REQUESTS
forms_service.forms().batchUpdate(
    formId=form_id,
    body={"requests": requests}
).execute()

print("✅ Form populated successfully!")
