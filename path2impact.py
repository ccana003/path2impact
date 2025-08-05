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
import time

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

enable_redcap_upload = st.checkbox("Upload processed PDFs to REDCap")

st.markdown("---")

# ==============================================
# HELPER FUNCTIONS
# ==============================================
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, "scoring_cache.csv")

# Initialize cache
if os.path.exists(CACHE_FILE):
    scoring_cache = pd.read_csv(CACHE_FILE)
else:
    scoring_cache = pd.DataFrame(columns=["File", "Principle", "Subcategory", "Score", "Rationale"])
    scoring_cache.to_csv(CACHE_FILE, index=False)


def extract_sections(text):
    """Extract PDF text into sections (from ai_script.py)"""
    sections = {
        "Introduction": "",
        "Methods": "",
        "Results": "",
        "Discussion": "",
        "Acknowledgments": "",
        "Funding": ""
    }
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
    """Extracts all text from PDF"""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def generate_prompt(principle_name, subcat, scoring, keywords, sections, where_to_search):
    """Generate GPT prompt for scoring"""
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
    """Retrieve cached score if available"""
    row = scoring_cache[
        (scoring_cache["File"] == file) &
        (scoring_cache["Principle"] == principle) &
        (scoring_cache["Subcategory"] == subcat)
    ]
    if len(row) > 0:
        return int(row.iloc[0]["Score"]), row.iloc[0]["Rationale"]
    return None, None


def save_to_cache(file, principle, subcat, score, rationale):
    """Save a new score to cache"""
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
    """Placeholder scoring for testing skip_ai mode"""
    return "Score: 1\nRationale: Placeholder scoring for testing."


# ==============================================
# 3. PROCESS PDFs
# ==============================================
if st.button("🚀 Run Analysis"):
    if not rubric_files:
        st.error("Please upload at least one rubric CSV.")
    else:
        # Collect PDFs
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

            # Load rubrics
            rubrics = []
            for rfile in rubric_files:
                df = pd.read_csv(rfile)
                principle_name = os.path.splitext(rfile.name)[0]
                rubrics.append((principle_name, df))

            full_results = []
            summary_results = []

            total_tasks = len(pdf_files) * sum(len(df) for _, df in rubrics)
            progress_bar = st.progress(0)
            log_placeholder = st.empty()
            task_count = 0

            for pdf_file in pdf_files:
                pdf_name = os.path.basename(pdf_file)
                text = extract_text_from_pdf(pdf_file)
                sections = extract_sections(text)

                for principle_name, rubric_df in rubrics:
                    total_score = 0
                    subcat_scores = []

                    for _, row in rubric_df.iterrows():
                        subcat = row["Subcategory"]
                        scoring = row["Scoring Criteria"]
                        keywords = row["Keywords"]
                        where = row["Where to Search"]

                        # Update log
                        log_placeholder.write(f"Processing {pdf_name} → {principle_name}:{subcat}")

                        score, rationale = get_cached_score(pdf_name, principle_name, subcat)

                        if score is None:
                            if skip_ai:
                                ai_output = dummy_openai_score("prompt")
                            else:
                                prompt = generate_prompt(principle_name, subcat, scoring, keywords, sections, where)
                                import openai
                                openai.api_key = openai_api_key
                                response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0
                                )
                                ai_output = response['choices'][0]['message']['content']

                            # Parse output
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

                        # Update progress
                        task_count += 1
                        progress_bar.progress(task_count / total_tasks)

                    # Save principle-level score
                    summary_results.append({
                        "File": pdf_name,
                        "Principle": principle_name,
                        "Total Score": total_score,
                        "Subcategory Breakdown": "; ".join(subcat_scores)
                    })

            # Save outputs
            full_df = pd.DataFrame(full_results)
            summary_df = pd.DataFrame(summary_results)

            full_csv = os.path.join(OUTPUT_DIR, "full_scoring_results.csv")
            summary_csv = os.path.join(OUTPUT_DIR, "principle_scores.csv")
            full_df.to_csv(full_csv, index=False)
            summary_df.to_csv(summary_csv, index=False)

            st.success(f"✅ Analysis complete! Results saved to {full_csv} and {summary_csv}")
            st.dataframe(summary_df)

            # -------------------------------
            # CSV Download Buttons
            # -------------------------------
            st.markdown("### ⬇️ Download Results")

            # Convert DataFrames to CSV for download
            full_csv_bytes = full_df.to_csv(index=False).encode('utf-8')
            summary_csv_bytes = summary_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Full Scoring Results (CSV)",
                data=full_csv_bytes,
                file_name="full_scoring_results.csv",
                mime="text/csv"
            )

            st.download_button(
                label="Download Principle Scores (CSV)",
                data=summary_csv_bytes,
                file_name="principle_scores.csv",
                mime="text/csv"
            )
            
            # -------------------------------
            # Cohen's Kappa Comparison (Optional)
            # -------------------------------
            st.markdown("---")
            st.header("📊 Step 4: Optional - Compare with Human Ratings")

            human_file = st.file_uploader("Upload human rater CSV (optional)", type=["csv"])
            
            if human_file:
                human_df = pd.read_csv(human_file)
                st.write("Preview of Human Ratings:")
                st.dataframe(human_df.head())

                # Merge AI and Human scores
                compare_df = pd.merge(
                    summary_df,
                    human_df,
                    on=["File", "Principle"],
                    how="inner"
                ).rename(columns={"Total Score": "AI Score"})

                st.subheader("Side-by-Side Comparison")
                st.dataframe(compare_df)

                # Compute Cohen's Kappa for each principle
                from sklearn.metrics import cohen_kappa_score

                kappa_results = []
                for principle in compare_df["Principle"].unique():
                    subset = compare_df[compare_df["Principle"] == principle]
                    kappa = cohen_kappa_score(subset["AI Score"], subset["Human Score"])
                    kappa_results.append({
                        "Principle": principle,
                        "Cohen's Kappa": round(kappa, 3)
                    })

                kappa_df = pd.DataFrame(kappa_results)
                st.subheader("Cohen's Kappa by Principle")
                st.dataframe(kappa_df)

                # Overall Kappa
                overall_kappa = cohen_kappa_score(compare_df["AI Score"], compare_df["Human Score"])
                st.success(f"Overall Cohen's Kappa: {round(overall_kappa, 3)}")

                # Optionally download comparison table
                compare_csv = compare_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download AI vs Human Comparison (CSV)",
                    data=compare_csv,
                    file_name="ai_vs_human_comparison.csv",
                    mime="text/csv"
                )
