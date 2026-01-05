import streamlit as st
import pickle
import re
import io
import pandas as pd

# optional parsers
try:
    import docx
except:
    docx = None

try:
    import pdfplumber
except:
    pdfplumber = None

# CONFIG 
MODEL_PATH = "best_pipeline.pkl"   
ALLOWED_TYPES = ["pdf", "docx", "txt"]

# If filename indicates internship, we force this label:
OVERRIDE_LABEL_FOR_INTERNS = "Internship"

# Keywords checked in filename (case-insensitive)
FNAME_INTERN_KEYS = ["intern", "internship", "trainee"]


st.set_page_config(page_title="Resume Classification", layout="wide")
st.title("Resume Predictor — Upload Resume")
st.write("Upload resumes (PDF / DOCX / TXT). App shows Name and Predicted Label.")


# Load trained model (NO DISPLAY MESSAGE)
@st.cache_data(show_spinner=False)
def load_pipeline(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)

pipeline, load_err = load_pipeline(MODEL_PATH)



# Text extraction 
def extract_text_from_docx_bytes(b):
    if docx is None:
        return ""
    try:
        doc = docx.Document(io.BytesIO(b))
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except:
        return ""

def extract_text_from_pdf_bytes(b):
    if pdfplumber is None:
        return ""
    try:
        text_list = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for page in pdf.pages:
                text_list.append(page.extract_text() or "")
        return "\n".join(text_list)
    except:
        return ""

def extract_text_from_txt_bytes(b):
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return str(b)

def extract_text(file):
    fname = file.name.lower()
    b = file.read()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(b)
    if fname.endswith(".docx"):
        return extract_text_from_docx_bytes(b)
    return extract_text_from_txt_bytes(b)


# Name extractor
def extract_name(text):
    if not text:
        return "Not Found"

    # pattern: Name: John Doe
    m = re.search(r"(?i)\bname[:\-\s]{1,6}([A-Z][A-Za-z ,.'\-]{1,80})", text)
    if m:
        return m.group(1).strip()

    # first plausible line
    for ln in text.splitlines()[:12]:
        s = ln.strip()
        if not s or len(s) > 80:
            continue
        parts = s.split()
        if 2 <= len(parts) <= 4 and not any(ch.isdigit() for ch in s):
            if re.search(r"(?i)\b(resume|curriculum|cv|objective|summary|profile)\b", s):
                continue
            return s

    return "Not Found"


# Filename-based internship 
def filename_indicates_internship(filename: str) -> bool:
    low = filename.lower()
    return any(kw in low for kw in FNAME_INTERN_KEYS)


# File Upload 
uploaded_files = st.file_uploader(
    "Upload resumes (PDF / DOCX / TXT) — multiple allowed",
    accept_multiple_files=True,
    type=ALLOWED_TYPES
)

if uploaded_files:
    rows = []
    texts = []

    for file in uploaded_files:
        txt = extract_text(file)
        nm = extract_name(txt)
        is_intern = filename_indicates_internship(file.name)

        rows.append({
            "filename": file.name,
            "name": nm,
            "is_intern": is_intern,
            "text": txt
        })

        texts.append(txt)

    # Predictions using model
    if pipeline is None:
        st.error("Model not loaded. Please place best_pipeline.pkl in app folder.")
        st.stop()

    try:
        raw_preds = pipeline.predict(texts)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Apply filename-based internship override
    final_labels = []
    for r, raw_label in zip(rows, raw_preds):
        if r["is_intern"]:
            final_labels.append(OVERRIDE_LABEL_FOR_INTERNS)
        else:
            final_labels.append(raw_label)

    # DataFrame 
    df = pd.DataFrame({
        "filename": [r["filename"] for r in rows],
        "name": [r["name"] for r in rows],
        "predicted_label": final_labels
    })

    # Show dataframe only
    st.markdown("### Results — Uploaded Resume Predictions")
    st.dataframe(df, use_container_width=True)

    # Allow CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions (CSV)",
        data=csv,
        file_name="resume_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload resumes to begin.")
