import streamlit as st
import os
import tempfile
import pandas as pd
import fitz  # PyMuPDF
import requests
import pdfplumber
import re
from bs4 import BeautifulSoup
import nltk
from datetime import datetime
from openai import OpenAI, OpenAIError
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

nltk.download('punkt')

# ==============================================
# STREAMLIT: PATH2IMPACT
# ==============================================
st.title("📊 Path2Impact - Principle Scoring Tool")

# -------------------------------
# 1. API Keys
# -------------------------------
st.header("🔑 Step 1: Enter API Keys")
openai_api_key = st.text_input("OpenAI API Key", type="password")
dimensions_api_key = st.text_input("Dimensions API Key", type="password")
redcap_token = st.text_input("REDCap API Token (optional)", type="password")
unpaywall_email = st.text_input("Unpaywall Email", type="default")

skip_ai = st.checkbox("Skip OpenAI scoring (use cached results if available)")

st.markdown("---")

# -------------------------------
# 2. Upload Rubrics & Documents
# -------------------------------
st.header("📄 Step 2: Upload Rubrics & PDFs")

rubric_files = st.file_uploader("Upload one or more rubric CSVs", type=["csv"], accept_multiple_files=True)

upload_option = st.radio("How would you like to provide your PDFs?",
                         ["Upload PDF files", "Specify a folder path"])

uploaded_files = []
folder_path = ""
pdf_files = []

if upload_option == "Upload PDF files":
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
else:
    folder_path = st.text_input("Enter the folder path containing PDFs")

st.markdown("---")

# ==============================================
# HELPER FUNCTIONS
# ==============================================
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, "scoring_cache.csv")

if os.path.exists(CACHE_FILE):
    scoring_cache = pd.read_csv(CACHE_FILE)
else:
    scoring_cache = pd.DataFrame(columns=["File", "Principle", "Subcategory", "Score", "Rationale"])
    scoring_cache.to_csv(CACHE_FILE, index=False)


def extract_sections(text):
    sections = {k: "" for k in [
        "Introduction", "Methods", "Results", "Discussion", "Acknowledgments", "Funding"
    ]}
    lines = text.split('\n')
    current = None
    for line in lines:
        l = line.strip().lower()
        if "introduction" in l:
            current = "Introduction"
        elif "methods" in l or "methodology" in l:
            current = "Methods"
        elif "results" in l:
            current = "Results"
        elif "discussion" in l:
            current = "Discussion"
        elif "acknowledgments" in l:
            current = "Acknowledgments"
        elif "funding" in l:
            current = "Funding"
        elif "references" in l or "conclusion" in l:
            current = None
        if current:
            sections[current] += line + "\n"
    return sections


def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return "".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def generate_prompt(principle_name, subcat, scoring, keywords, sections, where_to_search):
    search_parts = [s.strip() for s in where_to_search.split(";")]
    collected_text = ""
    for part in search_parts:
        if part in sections:
            collected_text += f"[{part}]\n{sections[part]}\n"

    return f"""
You are evaluating whether a research publication demonstrates the principle '{principle_name}',
focusing on subcategory '{subcat}'.

Rubric:
- Scoring: {scoring}
- Keywords: {keywords}

Relevant Sections:
{collected_text}

Task:
1. Review ONLY the above sections.
2. Determine if this subcategory is truly implemented.
3. Respond in the exact format:

Score: X
Rationale: <your reasoning>
    """.strip()


def get_cached_score(file, principle, subcat):
    row = scoring_cache[
        (scoring_cache["File"] == file) &
        (scoring_cache["Principle"] == principle) &
        (scoring_cache["Subcategory"] == subcat)
    ]
    if len(row) > 0:
        return int(row.iloc[0]["Score"]), row.iloc[0]["Rationale"]
    return None, None


def save_to_cache(file, principle, subcat, score, rationale):
    global scoring_cache
    new_row = pd.DataFrame([{
        "File": file,
        "Principle": principle,
        "Subcategory": subcat,
        "Score": score,
        "Rationale": rationale
    }])
    scoring_cache = pd.concat([scoring_cache, new_row], ignore_index=True)
    scoring_cache.to_csv(CACHE_FILE, index=False)


def dummy_openai_score(prompt):
    return "Score: 1\nRationale: Placeholder scoring for testing."


# ==============================================
# 3. PROCESS PDFs
# ==============================================
if st.button("🚀 Run Analysis"):
    if not rubric_files:
        st.error("Please upload at least one rubric CSV.")
    else:
        if upload_option == "Upload PDF files" and uploaded_files:
            tmp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                pdf_files.append(file_path)
        elif upload_option == "Specify a folder path" and os.path.isdir(folder_path):
            pdf_files = [os.path.join(folder_path, f)
                         for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

        if not pdf_files:
            st.error("No PDF files found. Please upload or provide a valid folder path.")
        else:
            st.success(f"Found {len(pdf_files)} PDF files. Starting analysis...")

            client = None
            if not skip_ai:
                client = OpenAI(api_key=openai_api_key)

            rubrics = [(os.path.splitext(rfile.name)[0], pd.read_csv(rfile)) for rfile in rubric_files]
            full_results, summary_results = [], []

            total_tasks = len(pdf_files) * sum(len(df) for _, df in rubrics)
            progress_bar, log_placeholder = st.progress(0), st.empty()
            task_count = 0

            for pdf_file in pdf_files:
                pdf_name = os.path.basename(pdf_file)
                text, sections = extract_text_from_pdf(pdf_file), None
                sections = extract_sections(text)

                for principle_name, rubric_df in rubrics:
                    total_score, subcat_scores = 0, []

                    for _, row in rubric_df.iterrows():
                        subcat, scoring, keywords, where = row["Subcategory"], row["Scoring Criteria"], row["Keywords"], row["Where to Search"]
                        log_placeholder.write(f"Processing {pdf_name} → {principle_name}:{subcat}")
                        score, rationale = get_cached_score(pdf_name, principle_name, subcat)

                        if score is None:
                            if skip_ai:
                                ai_output = dummy_openai_score("prompt")
                            else:
                                prompt = generate_prompt(principle_name, subcat, scoring, keywords, sections, where)
                                try:
                                    response = client.chat.completions.create(
                                        model="gpt-4-turbo",
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=0
                                    )
                                    ai_output = response.choices[0].message.content

                                except OpenAIError as e:
                                    st.error(f"OpenAI API call failed: {str(e)}")
                                    ai_output = "Score: 0\Ratioinale: API call failed due to quota or billing issue."

                            try:
                                lines = ai_output.strip().splitlines()
                                score_line = [l for l in lines if l.lower().startswith("score")][0]
                                rationale_line = [l for l in lines if l.lower().startswith("rationale")][0]
                                score = int(re.search(r'\d', score_line).group())
                                rationale = rationale_line.split(":", 1)[1].strip()
                            except:
                                score, rationale = 0, "Parsing error."

                            save_to_cache(pdf_name, principle_name, subcat, score, rationale)

                        total_score += score
                        subcat_scores.append(f"{subcat}:{score}")

                        full_results.append({
                            "File": pdf_name,
                            "Principle": principle_name,
                            "Subcategory": subcat,
                            "Score": score,
                            "Rationale": rationale
                        })

                        task_count += 1
                        progress_bar.progress(task_count / total_tasks)

                    summary_results.append({
                        "File": pdf_name,
                        "Principle": principle_name,
                        "Total Score": total_score,
                        "Subcategory Breakdown": "; ".join(subcat_scores)
                    })

            summary_df = pd.DataFrame(summary_results)
            st.session_state['ai_summary_df'] = summary_df.copy()
            pivot_df = summary_df.pivot(index="File", columns="Principle", values="Total Score").reset_index()
            st.session_state['ai_pivot_df'] = pivot_df.copy()

            st.success("✅ Analysis complete! Results stored in session state.")
            st.subheader("Summary (Wide Form: One Row Per PDF)")
            st.dataframe(pivot_df)

# ==============================================
# 4. COHEN'S KAPPA (Always Visible)
# ==============================================
st.markdown("---")
st.header("📊 Step 4: Optional - Compare with Human Ratings")

human_file = st.file_uploader("Upload human rater CSV (optional)", type=["csv"])

if human_file:
    if 'ai_summary_df' not in st.session_state:
        st.warning("⚠️ Please run AI analysis first before uploading human ratings.")
    else:
        human_df = pd.read_csv(human_file)
        st.write("Preview of Human Ratings:")
        st.dataframe(human_df.head())

        if 'File' not in human_df.columns and 'publication' in human_df.columns:
            human_df.rename(columns={'publication': 'File'}, inplace=True)

        if 'Principle' not in human_df.columns:
            human_long = human_df.melt(id_vars=['File'], var_name='Principle', value_name='Human Score')
        else:
            human_long = human_df.copy()
            if 'Human Score' not in human_long.columns:
                last_col = human_long.columns[-1]
                human_long.rename(columns={last_col: 'Human Score'}, inplace=True)

        human_long = human_long.dropna(subset=['Human Score'])
        human_long['File'] = human_long['File'].astype(str).str.strip().str.replace("\\", "/", regex=False).str.replace(" ", "_")
        summary_df = st.session_state['ai_summary_df'].copy()
        summary_df['File'] = summary_df['File'].astype(str).str.strip().str.replace("\\", "/", regex=False).str.replace(" ", "_")

        compare_df = pd.merge(summary_df, human_long, on=["File", "Principle"], how="inner").rename(columns={"Total Score": "AI Score"})
        st.session_state['compare_df'] = compare_df

        if not compare_df.empty:
            st.subheader("Side-by-Side Comparison")
            st.dataframe(compare_df)

            kappa_results = []
            for principle in compare_df["Principle"].unique():
                subset = compare_df[compare_df["Principle"] == principle]
                kappa = cohen_kappa_score(subset["AI Score"], subset["Human Score"])
                kappa_results.append({"Principle": principle, "Cohen's Kappa": round(kappa, 3)})

            kappa_df = pd.DataFrame(kappa_results)
            st.subheader("Cohen's Kappa by Principle")
            st.dataframe(kappa_df)

            overall_kappa = cohen_kappa_score(compare_df["AI Score"], compare_df["Human Score"])
            st.success(f"Overall Cohen's Kappa: {round(overall_kappa, 3)}")

            # Visualization
            st.subheader("📊 Cohen's Kappa Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(kappa_df["Principle"], kappa_df["Cohen's Kappa"], color='skyblue', label="Per Principle Kappa")
            ax.axhline(overall_kappa, color='red', linestyle='--', label=f"Overall Kappa ({overall_kappa:.3f})")
            ax.axhline(0.3, color='green', linestyle=':', label='Significance Threshold (0.3)')
            ax.set_ylabel("Cohen's Kappa")
            ax.set_xlabel("Principle")
            ax.set_ylim(-0.1, 1.0)
            ax.set_title("Cohen's Kappa Agreement by Principle")
            plt.xticks(rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)

        else:
            st.warning("⚠️ No overlapping File + Principle pairs found.")
            st.write("AI Files Example:", summary_df['File'].unique()[:5])
            st.write("Human Files Example:", human_long['File'].unique()[:5])
            st.write("AI Principles Example:", summary_df['Principle'].unique()[:5])
            st.write("Human Principles Example:", human_long['Principle'].unique()[:5])

            if st.checkbox("Show all non-matching files and principles"):
                unmatched_files = set(summary_df['File']) - set(human_long['File'])
                unmatched_principles = set(summary_df['Principle']) - set(human_long['Principle'])
                st.write("### 🔹 Files in AI results but not in human CSV:")
                st.write(unmatched_files if unmatched_files else "None")
                st.write("### 🔹 Principles in AI results but not in human CSV:")
                st.write(unmatched_principles if unmatched_principles else "None")
