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
