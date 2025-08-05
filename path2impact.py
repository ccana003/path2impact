import streamlit as st
import os
import tempfile
import pandas as pd
import fitz  # PyMuPDF
import requests
import re
import nltk
from datetime import datetime
from openai import OpenAI
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

nltk.download('punkt')

# ==============================================
# Streamlit Page Setup
# ==============================================
st.set_page_config(
    page_title="Path2Impact: Principle Scoring Tool",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .big-title {
        font-size: 38px !important;
        font-weight: 700;
        text-align: center;
        color: #2C3E50;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px !important;
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="display:flex; align-items:center; justify-content:center; margin-bottom:30px;">
        <img src="p2i_logo.png" style="width:80px; margin-right:20px;">
        <div style="text-align:left;">
            <div style="font-size:38px; font-weight:700; color:#2C3E50; line-height:1.2;">
                Translational Principle Scoring
            </div>
            <div style="font-size:20px; color:#7F8C8D; margin-top:5px;">
                Evaluate research publications against NCATS Translational Science Principles
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ==============================================
# 1. API Keys
# ==============================================
with st.expander("🔑 Step 1: Enter API Keys", expanded=True):
    st.info("💡 Enter your OpenAI and optional APIs. Leave blank to test with cached AI results.")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    unpaywall_email = st.text_input("Unpaywall Email (required for DOI fetching)", type="default")
    skip_ai = st.checkbox("Skip OpenAI scoring (use cached results if available)")

st.markdown("---")

# ==============================================
# 2. Upload Rubrics & PDFs
# ==============================================
with st.expander("📄 Step 2: Upload Rubrics & PDFs", expanded=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        rubric_files = st.file_uploader(
            "Upload one or more rubric CSVs",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload rubric files to define principles and subcategories."
        )

    with col2:
        st.info("💡 Tip: Upload multiple rubric files to evaluate multiple principles at once!")

    upload_option = st.radio(
        "How would you like to provide your PDFs?",
        ["Upload PDF files", "Specify a folder path"]
    )

    uploaded_files, folder_path, pdf_files = [], "", []

    if upload_option == "Upload PDF files":
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Drag and drop multiple PDFs here for scoring."
        )
    else:
        folder_path = st.text_input(
            "Enter the folder path containing PDFs",
            help="Example: PDFs or /workspaces/path2impact/PDFs in Codespaces"
        )

st.markdown("---")

# ==============================================
# 3. Optional: Download PDFs via Unpaywall
# ==============================================
with st.expander("📥 Step 3: (Optional) Download PDFs via Unpaywall"):
    st.info("💡 Upload a CSV with DOIs to automatically fetch available PDFs from Unpaywall.")

    doi_csv = st.file_uploader("Upload CSV with DOIs (optional)", type=["csv"])
    use_unpaywall = st.checkbox("Fetch missing PDFs from Unpaywall using DOIs")

    downloaded_dir = "Downloaded_PDFs"
    os.makedirs(downloaded_dir, exist_ok=True)
    downloaded_files = []

    if doi_csv and use_unpaywall:
        df_dois = pd.read_csv(doi_csv)
        doi_column = [col for col in df_dois.columns if col.lower() == 'doi']

        if not doi_column:
            st.error("❌ No 'doi' column found in uploaded CSV.")
        else:
            doi_column = doi_column[0]
            total_dois = len(df_dois[doi_column].dropna())
            st.write(f"Processing **{total_dois}** DOIs...")

            download_progress = st.progress(0)
            download_status = st.empty()

            for idx, doi in enumerate(df_dois[doi_column].dropna()):
                pdf_filename = doi.replace("/", "_") + ".pdf"
                pdf_path = os.path.join(downloaded_dir, pdf_filename)

                download_progress.progress((idx + 1) / total_dois)
                download_status.markdown(f"**📥 Fetching DOI:** `{doi}` ({idx+1}/{total_dois})")

                if os.path.exists(pdf_path):
                    downloaded_files.append(pdf_path)
                    continue

                api_url = f"https://api.unpaywall.org/v2/{doi}?email={unpaywall_email}"
                try:
                    r = requests.get(api_url, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    pdf_url = data.get('best_oa_location', {}).get('url_for_pdf')

                    if pdf_url:
                        pdf_resp = requests.get(pdf_url, timeout=15)
                        if pdf_resp.headers.get('Content-Type', '').lower() == 'application/pdf':
                            with open(pdf_path, 'wb') as f:
                                f.write(pdf_resp.content)
                            downloaded_files.append(pdf_path)
                            st.success(f"✅ {pdf_filename} downloaded.")
                        else:
                            st.warning(f"⚠️ {doi} has no direct PDF link.")
                    else:
                        st.warning(f"⚠️ No Open Access PDF for DOI {doi}.")
                except Exception as e:
                    st.error(f"❌ Failed to fetch {doi}: {e}")

            pdf_files.extend(downloaded_files)
            st.success(f"✅ Downloaded {len(downloaded_files)} PDFs from {total_dois} DOIs.")

st.markdown("---")

# ==============================================
# Helper Functions
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
# 4. Run Analysis
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
                text = extract_text_from_pdf(pdf_file)
                sections = extract_sections(text)

                for principle_name, rubric_df in rubrics:
                    total_score, subcat_scores = 0, []

                    for _, row in rubric_df.iterrows():
                        subcat, scoring, keywords, where = row["Subcategory"], row["Scoring Criteria"], row["Keywords"], row["Where to Search"]
                        log_placeholder.markdown(f"**📄 Processing:** `{pdf_name}` → **{principle_name}:{subcat}**")
                        score, rationale = get_cached_score(pdf_name, principle_name, subcat)

                        if score is None:
                            if skip_ai:
                                ai_output = dummy_openai_score("prompt")
                            else:
                                try:
                                    prompt = generate_prompt(principle_name, subcat, scoring, keywords, sections, where)
                                    response = client.chat.completions.create(
                                        model="gpt-4-turbo",
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=0
                                    )
                                    ai_output = response.choices[0].message.content
                                except Exception as e:
                                    st.error(f"OpenAI API call failed: {e}")
                                    ai_output = "Score: 0\nRationale: API call failed."

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
# 5. Cohen's Kappa Comparison
# ==============================================
st.markdown("---")
st.header("📊 Step 5: Optional - Compare with Human Ratings")

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
