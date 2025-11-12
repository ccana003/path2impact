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

# Banner
with st.container():
    col_logo, col_title = st.columns([2, 5])
    with col_logo:
        st.image("p2i_logo.png", width=100)
    with col_title:
        st.markdown(
            """
            <div style="font-size:38px; font-weight:700; color:#2C3E50; line-height:1.2;">
                Translational Principle Scoring
            </div>
            <div style="font-size:20px; color:#7F8C8D;">
                Evaluate research publications against NCATS Translational Science Principles
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================
# 1. API Keys & Run Mode
# ==============================================
with st.expander("🔑 Step 1: Enter API Keys / Run Settings", expanded=True):
    st.info("💡 Leave OpenAI blank to use the deterministic keyword-based scorer.")
    cols = st.columns(3)
    with cols[0]:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
    with cols[1]:
        unpaywall_email = st.text_input("Unpaywall Email (for DOI fetching)", type="default")
    with cols[2]:
        skip_ai = st.checkbox("Skip OpenAI (use keyword heuristic)", value=True,
                              help="Deterministic, repeatable 0/1/2 scoring based on rubric keywords.")
    cols2 = st.columns(3)
    with cols2[0]:
        bypass_cache = st.checkbox("Bypass cache for this run", value=True,
                                   help="Ignore any previously cached scores and recompute.")
    with cols2[1]:
        show_cache = st.checkbox("Show cache table after run", value=False)
    with cols2[2]:
        clear_now = st.button("🧹 Clear cache file now")

st.markdown("---")

# ==============================================
# 2. Upload Rubrics & PDFs
# ==============================================
with st.expander("📄 Step 2: Upload Rubrics & PDFs", expanded=False):
    col1, col2 = st.columns([4, 1])
    with col1:
        rubric_files = st.file_uploader(
            "2a: Upload one or more rubric CSVs",
            type=["csv"], accept_multiple_files=True,
            help="Upload rubric files to define principles and subcategories."
        )
    with col2:
        st.info("💡 Upload multiple rubric files to score multiple principles.")

    st.markdown("<br>", unsafe_allow_html=True)

    upload_option = st.radio(
        "2b: How would you like to provide your PDFs?",
        ["Upload PDF files", "Specify a folder path"]
    )

    uploaded_files, folder_path, pdf_files = [], "", []
    if upload_option == "Upload PDF files":
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"], accept_multiple_files=True,
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
with st.expander("📥 Step 3: (Optional) Download PDFs via Unpaywall", expanded=False):
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
                    r = requests.get(api_url, timeout=10); r.raise_for_status()
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

            st.success(f"✅ Downloaded {len(downloaded_files)} PDFs.")
            # Optionally add to working list
            # pdf_files.extend(downloaded_files)

st.markdown("---")

# ==============================================
# Helper Functions & Cache
# ==============================================
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, "scoring_cache.csv")

if clear_now:
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.success("🧹 Cache file deleted.")
    except Exception as e:
        st.error(f"Could not remove cache: {e}")

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
        elif ("methods" in l) or ("methodology" in l):
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
    search_parts = [s.strip() for s in str(where_to_search).split(";") if s.strip()]
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

def get_cached_score(file, principle, subcat, bypass=False):
    if bypass:
        return None, None
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

# --- Heuristic scorer (content-aware, deterministic) ---
def heuristic_score_from_keywords(keywords, where_to_search, sections):
    """
    Returns (score 0/1/2, rationale str).
    Scoring rule (simple but effective):
      - score 2: >=2 keyword hits across relevant sections, with at least 2 distinct keywords hit
      - score 1: >=1 keyword hit
      - score 0: else
    """
    # Collect relevant text
    search_parts = [s.strip() for s in str(where_to_search).split(";") if s.strip()]
    text = ""
    for part in search_parts:
        if part in sections:
            text += "\n" + sections[part]
    if not text:
        # fallback: whole doc if no specific section
        text = "\n".join(sections.values())

    text_low = text.lower()
    kw_list = [k.strip().lower() for k in str(keywords).split(";") if k.strip()]
    if not kw_list:
        return 0, "No keywords provided in rubric."

    hits = 0
    hit_terms = set()
    for kw in kw_list:
        # phrase hit or individual token hit
        if kw in text_low:
            hits += 1
            hit_terms.add(kw)
        else:
            # try token-level presence (simple)
            toks = [t for t in re.split(r"[^a-z0-9]+", kw) if t]
            if all(t in text_low for t in toks):
                hits += 1
                hit_terms.add(kw)

    if hits >= 2 and len(hit_terms) >= 2:
        return 2, f"Found multiple keyword hits ({len(hit_terms)}) across relevant sections."
    if hits >= 1:
        return 1, f"Found at least one keyword hit: {sorted(list(hit_terms))[:3]}"
    return 0, "No rubric keywords detected in relevant sections."

def dummy_openai_score(prompt):
    # kept for completeness; not used when heuristic is enabled
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
            if not skip_ai and openai_api_key:
                client = OpenAI(api_key=openai_api_key)

            # Load rubrics and track subcategory counts per principle (for correct max normalization)
            rubrics = []
            principle_subcat_counts = {}
            for rfile in rubric_files:
                pname = os.path.splitext(rfile.name)[0]  # e.g., creativity_innovation_rubric
                df = pd.read_csv(rfile)
                rubrics.append((pname, df))
                principle_subcat_counts[pname] = len(df)

            # Make available in session (used in Step 7)
            st.session_state['principle_subcat_counts_raw'] = principle_subcat_counts.copy()

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
                        subcat = row.get("Subcategory", "")
                        scoring = row.get("Scoring Criteria", "")
                        keywords = row.get("Keywords", "")
                        where = row.get("Where to Search", "")

                        log_placeholder.markdown(f"**📄 Processing:** `{pdf_name}` → **{principle_name}:{subcat}**")

                        # Cache
                        score, rationale = get_cached_score(pdf_name, principle_name, subcat, bypass=bypass_cache)

                        if score is None:
                            if skip_ai or not client:
                                # deterministic heuristic
                                score, rationale = heuristic_score_from_keywords(keywords, where, sections)
                            else:
                                # OpenAI path
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

                                if not (skip_ai or not client):
                                    # parse OpenAI output
                                    try:
                                        lines = ai_output.strip().splitlines()
                                        score_line = [l for l in lines if l.lower().startswith("score")][0]
                                        rationale_line = [l for l in lines if l.lower().startswith("rationale")][0]
                                        score = int(re.search(r'\d', score_line).group())
                                        rationale = rationale_line.split(":", 1)[1].strip()
                                    except Exception:
                                        score, rationale = 0, "Parsing error."

                            save_to_cache(pdf_name, principle_name, subcat, score, rationale)

                        total_score += int(score)
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

            if show_cache and os.path.exists(CACHE_FILE):
                st.caption("Cache preview (tail):")
                try:
                    _tmp = pd.read_csv(CACHE_FILE)
                    st.dataframe(_tmp.tail(20))
                except Exception as e:
                    st.info(f"Couldn't read cache: {e}")

# ==============================================
# 5 & 6. Cohen's Kappa Comparison (Unweighted + Weighted)
# ==============================================
st.markdown("---")
st.header("📊 Step 5 & 6: AI vs Human Ratings (Cohen's Kappa)")

human_file = st.file_uploader("Upload human rater CSV (optional)", type=["csv"])

if human_file:
    if 'ai_summary_df' not in st.session_state:
        st.warning("⚠️ Please run AI analysis first before uploading human ratings.")
    else:
        import numpy as np
        from sklearn.metrics import cohen_kappa_score
        import matplotlib.pyplot as plt

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

        human_long['File'] = (
            human_long['File'].astype(str)
            .str.strip()
            .str.replace("\\", "/", regex=False)
            .str.replace(" ", "_")
        )
        summary_df = st.session_state['ai_summary_df'].copy()
        summary_df['File'] = (
            summary_df['File'].astype(str)
            .str.strip()
            .str.replace("\\", "/", regex=False)
            .str.replace(" ", "_")
        )

        compare_df = pd.merge(
            summary_df,
            human_long,
            on=["File", "Principle"],
            how="inner"
        ).rename(columns={"Total Score": "AI Score"})

        st.session_state['compare_df'] = compare_df

        if not compare_df.empty:
            st.subheader("Side-by-Side Comparison")
            st.dataframe(compare_df)

            kappa_results = []
            for principle in compare_df["Principle"].unique():
                subset = compare_df[compare_df["Principle"] == principle]

                unweighted = cohen_kappa_score(subset["AI Score"], subset["Human Score"])
                if np.isnan(unweighted):
                    unweighted = 1.0 if (subset["AI Score"].values == subset["Human Score"].values).all() else 0.0

                weighted = cohen_kappa_score(subset["AI Score"], subset["Human Score"], weights="quadratic")
                if np.isnan(weighted):
                    weighted = 1.0 if (subset["AI Score"].values == subset["Human Score"].values).all() else 0.0

                kappa_results.append({
                    "Principle": principle,
                    "Unweighted Kappa": round(unweighted, 3),
                    "Weighted Kappa": round(weighted, 3)
                })

            kappa_df = pd.DataFrame(kappa_results)
            st.subheader("📊 Cohen's Kappa (Unweighted vs Weighted)")
            st.dataframe(kappa_df)

            overall_kappa = cohen_kappa_score(compare_df["AI Score"], compare_df["Human Score"])
            if np.isnan(overall_kappa):
                overall_kappa = 1.0 if (compare_df["AI Score"].values == compare_df["Human Score"].values).all() else 0.0

            overall_weighted = cohen_kappa_score(compare_df["AI Score"], compare_df["Human Score"], weights="quadratic")
            if np.isnan(overall_weighted):
                overall_weighted = 1.0 if (compare_df["AI Score"].values == compare_df["Human Score"].values).all() else 0.0

            st.success(f"Overall Cohen's Kappa: {round(overall_kappa, 3)} | Weighted: {round(overall_weighted, 3)}")

            with st.container():
                st.subheader("📊 Kappa Scores Visualization (Compact)")
                fig, ax = plt.subplots(figsize=(6, 3))
                x = range(len(kappa_df)); width = 0.35
                ax.bar([i - width/2 for i in x], kappa_df["Unweighted Kappa"], width, label='Unweighted', color='skyblue')
                ax.bar([i + width/2 for i in x], kappa_df["Weighted Kappa"], width, label='Weighted', color='orange')
                ax.axhline(overall_kappa, color='red', linestyle='--', label=f"Overall Unweighted ({overall_kappa:.3f})")
                ax.axhline(overall_weighted, color='purple', linestyle='--', label=f"Overall Weighted ({overall_weighted:.3f})")
                ax.set_ylabel("Kappa Score"); ax.set_xlabel("Principle"); ax.set_ylim(-0.1, 1.0)
                ax.set_title("AI vs Human Agreement by Principle")
                ax.set_xticks(list(x)); ax.set_xticklabels(kappa_df["Principle"], rotation=45, ha='right')
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("⚠️ No overlapping File + Principle pairs found.")
            st.write("AI Files Example:", summary_df['File'].unique()[:5])
            st.write("Human Files Example:", human_long['File'].unique()[:5])
            st.write("AI Principles Example:", summary_df['Principle'].unique()[:5])
            st.write("Human Principles Example:", human_long['Principle'].unique()[:5])

# ==============================================
# 7. Year-over-Year Insights & Translational Polygon
# ==============================================
import os as _os, sys as _sys, subprocess as _subp, numpy as _np

st.markdown("---")
st.header("📈 Step 7: Year-over-Year Insights & Translational Polygon")

if 'ai_pivot_df' not in st.session_state:
    st.warning("⚠️ Please run the analysis first to generate results.")
else:
    pivot_df = st.session_state['ai_pivot_df'].copy()

    # --- Alias rubric filenames -> canonical octagon labels
    alias_map = {
        "creativity_innovation_rubric": "Creativity & Innovation",
        "focus_on_unmet_needs_rubric": "Focus on Unmet Needs",
        "generalizable_solutions_rubric": "Generalizable Solutions",
        "cross_disciplinary_team_science_rubric": "Cross-disciplinary Team Science",
        "efficiency_and_speed_rubric": "Efficiency and Speed",
        "boundary_crossing_partnerships_rubric": "Boundary-crossing Partnerships",
        "bold_rigorous_research_approaches_rubric": "Bold & Rigorous Approaches",
        "deia_rubric": "Diversity, Equity, and Accessibility",
    }
    pivot_df.rename(columns={k: v for k, v in alias_map.items() if k in pivot_df.columns}, inplace=True)

    # --- Per-principle max = 2 × (# subcategories)
    # Read counts captured in Step 4; map to aliased names too
    subcat_counts_raw = st.session_state.get('principle_subcat_counts_raw', {})  # keys like rubric filenames
    principle_max_map = {}
    for raw_name, count in subcat_counts_raw.items():
        canonical = alias_map.get(raw_name, raw_name)
        principle_max_map[canonical] = max(1, int(count) * 2)

    # Derive Year from filename if missing
    if 'Year' not in pivot_df.columns and 'File' in pivot_df.columns:
        year_guess = pivot_df['File'].astype(str).str.extract(r'((?:19|20)\d{2})', expand=False)
        pivot_df['Year'] = pd.to_numeric(year_guess, errors='coerce')

    # Save a canonical CSV (optional external use)
    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "analysis_results.csv")
    pivot_df.to_csv(csv_path, index=False)
    st.success(f"Results saved to {csv_path}")

    # Identify principle columns (numeric only)
    principle_cols = [c for c in pivot_df.columns if c not in ("File", "Year")]
    for c in principle_cols:
        pivot_df[c] = pd.to_numeric(pivot_df[c], errors="coerce")
    principle_cols = [c for c in principle_cols if pivot_df[c].notna().any()]

    if len(principle_cols) < 3:
        st.warning("Need at least 3 principle columns to draw the polygon. "
                   "Upload more rubric CSVs (principles) or verify the alias map matches your columns.")
        st.stop()

    has_year = 'Year' in pivot_df.columns and pivot_df['Year'].notna().any()
    if has_year:
        yearly = (
            pivot_df[['Year'] + principle_cols]
            .dropna(subset=principle_cols, how="all")
            .groupby('Year', as_index=False)
            .mean(numeric_only=True)
            .sort_values('Year')
        )
        yearly['Year'] = pd.to_numeric(yearly['Year'], errors='coerce')
        yearly = yearly[yearly['Year'].between(1990, 2035)]
    else:
        yearly = pd.DataFrame([{**{'Year': 'All'}, **pivot_df[principle_cols].mean(numeric_only=True).to_dict()}])

    if yearly.empty:
        st.warning("No usable yearly data after filtering. Check that your filenames include years or add a 'Year' column.")
        st.stop()

    # === Controls
    st.subheader("Data sanity & scaling")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        use_per_principle_norm = st.checkbox("Use per-principle max (2×subcats)", value=True,
                                             help="Recommended: normalizes each principle by its true maximum.")
    with c2:
        boost = st.slider("Contrast boost (visual only)", 1.0, 3.0, 1.0, 0.1)
    with c3:
        jitter_amp = st.slider("Jitter amplitude (±)", 0.00, 0.30, 0.00, 0.01,
                               help="Adds small noise to confirm visuals respond.")

    # === Normalize for visuals
    norm = yearly.copy()
    for c in principle_cols:
        if use_per_principle_norm and (c in principle_max_map):
            denom = float(principle_max_map[c])
        else:
            # fallback: scale by observed max if we don't have a count
            denom = float(max(1.0, yearly[c].max()))
        norm[c] = _np.clip(yearly[c] / denom, 0, 1.0)

    if jitter_amp > 0 and len(norm) > 0:
        rng = _np.random.default_rng(42)
        for c in principle_cols:
            norm[c] = _np.clip(norm[c] + rng.uniform(-jitter_amp, jitter_amp, size=len(norm)), 0, 1.0)

    for c in principle_cols:
        norm[c] = _np.clip(norm[c] * boost, 0, 1.0)

    # Invariance detection
    diffs = norm[principle_cols].diff().abs().sum().sum()
    all_equal = (diffs == 0)

    # === Data provenance / cache
    with st.expander("ℹ️ Data provenance & normalization details", expanded=False):
        st.write(f"Principles used: {principle_cols}")
        st.write("Per-principle max (2×subcat counts):", principle_max_map)

    # === Heatmap
    st.subheader("Heatmap: Average principle scores by year (normalized 0–1)")
    try:
        import plotly.graph_objs as go  # noqa
    except ModuleNotFoundError:
        st.warning("Plotly not installed. Installing once...")
        _subp.run([_sys.executable, "-m", "pip", "install", "plotly>=5.18.0"], check=True)
        st.rerun()
    import plotly.graph_objs as go

    heat = go.Figure(data=go.Heatmap(
        z=norm[principle_cols].values,
        x=principle_cols,
        y=norm['Year'].astype(str).tolist(),
        zmin=0, zmax=1,
        colorbar=dict(title="Norm score")
    ))
    heat.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=220 + 15*len(norm),
        title="Per-year normalized means (rows) by principle (columns)"
    )
    st.plotly_chart(heat, use_container_width=True)

    # === Change first → last year
    st.subheader("Change from first → last year by principle")
    if len(norm) >= 2 and has_year:
        first_row = norm.iloc[0]; last_row = norm.iloc[-1]
        delta = pd.Series({c: float(last_row[c] - first_row[c]) for c in principle_cols}).sort_values()
        delta_df = pd.DataFrame({
            "Principle": delta.index,
            "Δ (last - first)": delta.values,
            "% change (of first)": [ (d / (first_row[p] or 1e-6)) * 100.0 for p, d in zip(delta.index, delta.values) ]
        })
        st.dataframe(delta_df, hide_index=True)
    else:
        st.info("Only one year present. Showing overall means above.")

    # === Centroid drift
    st.subheader("Centroid drift over time")
    positions = {
        "Creativity & Innovation":              ( 1.00,  0.00),
        "Focus on Unmet Needs":                 ( 0.71,  0.71),
        "Generalizable Solutions":              ( 0.00,  1.00),
        "Cross-disciplinary Team Science":      (-0.71,  0.71),
        "Efficiency and Speed":                 (-1.00,  0.00),
        "Boundary-crossing Partnerships":       (-0.71, -0.71),
        "Bold & Rigorous Approaches":           ( 0.00, -1.00),
        "Diversity, Equity, and Accessibility": ( 0.71, -0.71),
    }
    ordered = [p for p in positions if p in principle_cols]
    st.caption(f"Principles used for centroid computation: {ordered if ordered else '— none —'}")

    def _centroid_xy(row):
        pts = []
        for label in ordered:
            val = row.get(label, _np.nan)
            if pd.notna(val):
                bx, by = positions[label]
                pts.append((bx * float(val), by * float(val)))
        if not pts:
            return None
        xs, ys = zip(*pts)
        return float(_np.mean(xs)), float(_np.mean(ys))

    centroids = []
    for _, r in norm.iterrows():
        c = _centroid_xy(r)
        if c is not None:
            centroids.append((r["Year"], c[0], c[1]))

    if len(centroids) < 2:
        st.info("Need at least two valid years with data to compute drift.")
        if centroids:
            st.dataframe(pd.DataFrame(centroids, columns=["Year", "Cx", "Cy"]), hide_index=True)
    else:
        cent_df = pd.DataFrame(centroids, columns=["Year", "Cx", "Cy"])
        cent_df["StepDist"] = _np.sqrt((cent_df["Cx"].diff()**2) + (cent_df["Cy"].diff()**2))
        st.dataframe(cent_df[["Year", "Cx", "Cy", "StepDist"]], hide_index=True)

    # === Translational polygon (animated)
    st.subheader("Translational polygon")
    if len(ordered) < 3:
        st.info(f"Not enough principle columns to draw a polygon (need ≥3). Found: {ordered}")
    else:
        # hide polygon if all rows equal (unless jitter makes differences)
        all_equal = (norm[principle_cols].diff().abs().sum().sum() == 0)
        if all_equal and jitter_amp == 0:
            st.info("Data invariant across years. Polygon animation is hidden to avoid implying change. "
                    "Use jitter to verify animation or run a non-uniform analysis.")
        else:
            def polygon_xy(row):
                xs, ys = [], []
                for label in ordered:
                    bx, by = positions[label]
                    mag = float(row[label])
                    xs.append(bx * mag); ys.append(by * mag)
                xs.append(xs[0]); ys.append(ys[0])
                return xs, ys

            bx = [positions[l][0] for l in ordered] + [positions[ordered[0]][0]]
            by = [positions[l][1] for l in ordered] + [positions[ordered[0]][1]]

            init_row = norm.iloc[0]
            init_x, init_y = polygon_xy(init_row)
            poly_trace = go.Scatter(x=init_x, y=init_y, mode="lines", fill="toself",
                                    line=dict(width=3), fillcolor="rgba(31, 119, 180, 0.25)",
                                    name=f"{init_row['Year']}")

            boundary_trace = go.Scatter(x=bx, y=by, mode="lines",
                                        line=dict(color="#999", width=1),
                                        name="Boundary", hoverinfo="skip")
            ring_traces = [
                go.Scatter(x=[x*r for x in bx], y=[y*r for y in by],
                           mode="lines", line=dict(color="#e5e5e5", width=1),
                           hoverinfo="skip", showlegend=False, name=f"r{r}")
                for r in (0.25, 0.5, 0.75, 1.0)
            ]
            label_trace = go.Scatter(
                x=[positions[p][0] for p in ordered], y=[positions[p][1] for p in ordered],
                mode="markers+text", marker=dict(size=6, color="#333"),
                text=ordered, textposition="top center", name="Principles", hoverinfo="skip"
            )

            frames = []
            for _, r in norm.iterrows():
                px, py = polygon_xy(r)
                frames.append(go.Frame(
                    name=str(r['Year']),
                    data=[go.Scatter(x=px, y=py, mode="lines", fill="toself",
                                     line=dict(width=3), fillcolor="rgba(31,119,180,0.25)",
                                     name=f"{r['Year']}")],
                    traces=[0],
                    layout=go.Layout(title_text=f"Translational Octagon — {r['Year']}")
                ))

            fig = go.Figure(
                data=[poly_trace, boundary_trace, *ring_traces, label_trace],
                layout=go.Layout(
                    showlegend=False,
                    xaxis=dict(range=[-1.2, 1.2], zeroline=False, visible=False),
                    yaxis=dict(range=[-1.2, 1.2], zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
                    margin=dict(l=10, r=10, t=40, b=10),
                    title=f"Translational Octagon — {norm.iloc[0]['Year']}",
                    updatemenus=[dict(
                        type="buttons", showactive=False, x=0.05, y=1.15, xanchor="left",
                        buttons=[
                            dict(label="▶ Play", method="animate",
                                 args=[None, {"frame": {"duration": 700, "redraw": True},
                                              "fromcurrent": True, "transition": {"duration": 250}}]),
                            dict(label="⏸ Pause", method="animate",
                                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate"}]),
                        ],
                    )],
                    sliders=[dict(
                        active=0, x=0.15, y=1.10, xanchor="left", len=0.75,
                        currentvalue={"visible": True, "prefix": "Year: ", "xanchor": "right"},
                        steps=[
                            dict(method="animate",
                                 args=[[str(y)], {"mode": "immediate",
                                                  "frame": {"duration": 0, "redraw": True},
                                                  "transition": {"duration": 150}}],
                                 label=str(y))
                            for y in norm['Year'].tolist()
                        ]
                    )],
                ),
                frames=frames
            )
            st.plotly_chart(fig, use_container_width=True)
