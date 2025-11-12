import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# 1. Plotting Function
# =========================================================
def plot_translational_polygon(yearly_scores, year_labels, principle_labels,
                               title="Translational Principles Over Time", save_path=None):
    """
    Plots a radar (spider) chart for translational principle scores across multiple years.
    Saves high-resolution images if save_path is provided.
    """
    num_vars = len(principle_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each year's polygon
    for i, scores in enumerate(yearly_scores):
        scores = scores + scores[:1]  # Close the polygon
        ax.plot(angles, scores, linewidth=2, label=year_labels[i])
        ax.fill(angles, scores, alpha=0.25)

    # Configure radar chart aesthetics
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(principle_labels, fontsize=10)

    max_score = max(max(s) for s in yearly_scores)
    ax.set_ylim(0, max_score)
    ax.set_rlabel_position(0)

    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()

    # Save high-resolution figure if requested
    if save_path:
        base, ext = os.path.splitext(save_path)
        for fmt in ["png", "svg", "pdf"]:
            out_file = f"{base}.{fmt}"
            plt.savefig(out_file, dpi=600, bbox_inches='tight')
            print(f"Saved: {out_file}")

    plt.close(fig)  # Close figure to avoid Streamlit auto-render


# =========================================================
# 2. CSV -> Polygon Generator
# =========================================================
def generate_polygon_from_results(csv_file, year_column="Year", score_columns=None, save_path=None):
    
    
    """
    Reads Path2Impact output CSV, aggregates scores by year, and generates a radar chart.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV from 'Run Analysis'.
    year_column : str
        Column name in CSV that indicates the year of publication.
    score_columns : list
        List of columns corresponding to principle scores (aggregated per year).
        If None, numeric columns will be auto-detected.
    save_path : str
        Path to save the high-resolution figure (without extension).
    """
    df = pd.read_csv(csv_file)

    # Determine numeric score columns
    score_columns = df.select_dtypes(include="number").columns.tolist()

    # Build per-year summary ONLY if a valid year column was provided
    yearly_data = None
    if year_column and year_column in df.columns:
        # Drop the year column from the set of columns to aggregate
        score_columns = [c for c in score_columns if c != year_column]

        if score_columns:  # only aggregate if we actually have score columns
            yearly_data = (
                df.groupby(year_column, as_index=False)[score_columns]
                .mean(numeric_only=True)
            )

    else:
        yearly_data = None  # skip year trend inside this function


    # Ensure Year is numeric if present
    if year_column in df.columns:
        df[year_column] = pd.to_numeric(df[year_column], errors='coerce')

    # Auto-detect score columns if not provided
    if score_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        score_columns = [c for c in numeric_cols if c != year_column]

    if not score_columns:
        raise ValueError("No numeric score columns found for polygon generation.")

    print(f"🔹 Using the following columns for the polygon: {score_columns}")

    # Group by year and calculate mean scores per principle
    yearly_data = df.groupby(year_column)[score_columns].mean(numeric_only=True)

    # Drop rows where year could not be detected
    yearly_data = yearly_data.dropna().reset_index()

    years = yearly_data[year_column].astype(int).tolist()
    scores = yearly_data[score_columns].values.tolist()

    # Create radar chart
    plot_translational_polygon(
        yearly_scores=scores,
        year_labels=[str(y) for y in years],
        principle_labels=score_columns,
        title="Institutional Translational Principle Scores by Year",
        save_path=save_path
    )

# =========================================================
# 3. OCTAGON (Cartesian) — drift/growth by year
# =========================================================
def generate_octagon_from_results(
    csv_file,
    year_column="Year",
    save_path=None,
    principle_order=None,
    principle_caps=None,
    principle_aliases=None,
    min_per_year=1,           # require ≥ this many papers per year to include that year
    year_min=1990,            # plausible year floor (adjust as you like)
    year_max=2035,            # plausible year ceiling (adjust as you like)
):
    """
    Octagon (Cartesian) visualization of principle scores with year-by-year drift.
    - Normalizes by fixed caps (default 0..6) so size is comparable across years.
    - Filters implausible years and years with too few papers.
    - Draws reference rings and a centroid drift line (start vs end highlighted).
    - Always returns a dict of saved files ({} if nothing rendered).
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    out_paths = {}  # always return a dict

    # ---------- Load & optional aliasing ----------
    df = pd.read_csv(csv_file)
    if principle_aliases:
        rename_map = {k: v for k, v in principle_aliases.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

    # ---------- Canonical octagon order ----------
    if principle_order is None:
        principle_order = [
            "Creativity & Innovation",
            "Focus on Unmet Needs",
            "Generalizable Solutions",
            "Cross-disciplinary Team Science",
            "Efficiency and Speed",
            "Boundary-crossing Partnerships",
            "Bold & Rigorous Approaches",
            "Diversity, Equity, and Accessibility",
        ]

    principle_cols = [c for c in principle_order if c in df.columns]
    if len(principle_cols) < 3:
        return out_paths  # need at least a triangle

    # Coerce to numeric; drop rows that are all-NaN across principles
    for c in principle_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=principle_cols, how="all")
    if df.empty:
        return out_paths

    # ---------- Year handling & filtering ----------
    has_year = year_column in df.columns
    if has_year:
        df[year_column] = pd.to_numeric(df[year_column], errors="coerce")
        # plausible range filter (prevents stray 2066s, etc.)
        df = df[(df[year_column] >= year_min) & (df[year_column] <= year_max)]
        # require minimum sample size per year
        if min_per_year > 1:
            keep = df.groupby(year_column).size()
            keep = keep[keep >= min_per_year].index
            df = df[df[year_column].isin(keep)]
        has_year = df[year_column].notna().any()

    # Aggregate: by year if present, else single "All"
    if has_year:
        yearly = (
            df[[year_column] + principle_cols]
            .groupby(year_column, as_index=False)
            .mean(numeric_only=True)
            .sort_values(year_column)
        )
    else:
        mean_vals = df[principle_cols].mean(numeric_only=True)
        yearly = pd.DataFrame([{**{year_column: "All"}, **mean_vals.to_dict()}])

    if yearly.empty:
        return out_paths

    # ---------- Normalization (fixed caps = stable across years) ----------
    if principle_caps is None:
        principle_caps = {c: 6 for c in principle_order}  # default rubric cap 0..6
    norm = yearly.copy()
    for col in principle_cols:
        cap = max(1.0, float(principle_caps.get(col, 6)))
        norm[col] = np.clip(norm[col] / cap, 0, 1.0)

    # ---------- Octagon geometry ----------
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

    def polygon_xy(row):
        xs, ys = [], []
        for label in ordered:
            bx, by = positions[label]
            mag = float(row[label])  # 0..1
            xs.append(bx * mag); ys.append(by * mag)
        xs.append(xs[0]); ys.append(ys[0])  # close
        return xs, ys

    # Precompute boundary once (unit shape)
    boundary_x = [positions[l][0] for l in ordered] + [positions[ordered[0]][0]]
    boundary_y = [positions[l][1] for l in ordered] + [positions[ordered[0]][1]]

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Boundary
    ax.plot(boundary_x, boundary_y, color="#999", linewidth=1)

    # Reference rings & radial axes (make growth/shrink obvious)
    for r in (0.25, 0.5, 0.75, 1.0):
        ax.plot([x * r for x in boundary_x], [y * r for y in boundary_y],
                color="#e5e5e5", linewidth=0.8)
    for lbl in ordered:
        px, py = positions[lbl]
        ax.plot([0, px], [0, py], color="#eeeeee", linewidth=0.8)

    # Vertex labels
    ax.scatter([positions[p][0] for p in ordered],
               [positions[p][1] for p in ordered], s=30, color="#333")
    for p in ordered:
        px, py = positions[p]
        ax.text(px, py, p, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, lw=0))

    # Colors per year token (keep insertion order)
    yrs = norm[year_column].tolist()
    uniq = list(dict.fromkeys(yrs))
    cmap = plt.get_cmap("tab10")
    color_for = {y: cmap(i % 10) for i, y in enumerate(uniq)}

    # Plot polygons & compute centroids
    centers = []
    for _, row in norm.iterrows():
        token = row[year_column]
        xs, ys = polygon_xy(row)
        ax.plot(xs, ys, color=color_for[token], linewidth=2, label=str(token))
        ax.fill(xs, ys, color=color_for[token], alpha=0.20)

        # centroid for drift path
        vx = []; vy = []
        for label in ordered:
            bx, by = positions[label]
            mag = float(row[label])
            vx.append(bx * mag); vy.append(by * mag)
        centers.append((token, sum(vx) / len(vx), sum(vy) / len(vy)))

    # Centroid drift line (start → end)
    if centers:
        centers_sorted = sorted(centers, key=lambda t: str(t[0]))
        ax.plot([c[1] for c in centers_sorted],
                [c[2] for c in centers_sorted],
                color="#333", linestyle="--", linewidth=1.2, alpha=0.85, label="Centroid drift")
        first = centers_sorted[0]; last = centers_sorted[-1]
        ax.scatter([first[1], last[1]], [first[2], last[2]], s=40,
                   color=["#2ca02c", "#d62728"], zorder=5)
        ax.text(first[1], first[2], f"{first[0]}", fontsize=8, ha="right", va="bottom")
        ax.text(last[1],  last[2],  f"{last[0]}",  fontsize=8, ha="left",  va="bottom")

    # Aesthetics
    ax.set_title("Translational Octagon — Drift/Growth by Year", pad=12)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.00))
    plt.tight_layout()

    # Save files
    if save_path:
        base, _ = os.path.splitext(save_path)
        for fmt in ("png", "svg", "pdf"):
            out_file = f"{base}.{fmt}"
            plt.savefig(out_file, dpi=600, bbox_inches="tight")
            out_paths[fmt] = out_file
            print(f"Saved: {out_file}")

    plt.close(fig)
    return out_paths
