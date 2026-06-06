import squarify
import pandas as pd
from pathlib import Path
import json

def export_treemap_tikz(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    out_path: str,
    width: int = 120,
    height: int = 80
):
    """
    Generate TikZ code for a squarified treemap.

    df         : DataFrame with label and value columns
    label_col  : Column name for labels
    value_col  : Column name for values (sizes)
    out_path   : Output .tex file path
    width/height : TikZ virtual canvas dimensions
    """

    labels = df[label_col].tolist()
    values = df[value_col].tolist()

    # Normalize sizes and generate rectangles
    sizes = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(sizes, 0, 0, width, height)

    # Color palette for categories
    colors = [
        "blue!60", "red!60", "green!60", "orange!60",
        "violet!60", "cyan!60", "pink!60", "gray!60",
        "yellow!60", "purple!60", "teal!60", "magenta!60",
        "lime!60", "olive!60", "brown!60", "black!40"
    ]

    lines = []
    lines.append(r"\begin{tikzpicture}[x=0.08cm,y=0.08cm]")
    lines.append(r"% Auto-generated squarified treemap")
    lines.append(rf"% Data: {len(labels)} items, total value: {sum(values):,}")

    for i, (r, label, val) in enumerate(zip(rects, labels, values)):
        x1, y1 = r["x"], r["y"]
        x2, y2 = r["x"] + r["dx"], r["y"] + r["dy"]
        color = colors[i % len(colors)]

        # Draw rectangle
        lines.append(
            rf"\fill[{color}] ({x1:.2f},{y1:.2f}) rectangle ({x2:.2f},{y2:.2f});"
        )

        # Add label only if area is large enough (avoid clutter)
        area = r["dx"] * r["dy"]
        min_area = (width * height * 0.015)  # 1.5% of total area

        if area > min_area:
            cx, cy = x1 + r["dx"]/2, y1 + r["dy"]/2
            # Use smaller font for labels
            lines.append(
                rf"\node[align=center,font=\scriptsize] at ({cx:.2f},{cy:.2f}) {{{label}}};"
            )

    lines.append(r"\end{tikzpicture}")

    # Write to file
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f" TikZ treemap exported to {out_path}")


def generate_iptc_treemaps(data_dir: Path = None, results_dir: Path = None):
    """
    Generate treemaps for all 16 IPTC categories.
    Each treemap shows GDELT themes mapped to that IPTC category,
    sized by total document count.
    """

    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"

    # Load IPTC mapping
    with open(results_dir / "gdelt_iptc_mapping_v2.json", 'r', encoding='utf-8') as f:
        iptc_data = json.load(f)

    iptc_mapping = pd.DataFrame(iptc_data['themes'])[['theme_code', 'iptc_final_id', 'iptc_final_label']]
    iptc_mapping = iptc_mapping.rename(columns={
        'iptc_final_id': 'iptc_id',
        'iptc_final_label': 'iptc_category'
    })

    # Load total docs data (from analysis.py output or directly from monthly)
    try:
        # Try to load from analysis output first
        analysis_dir = Path(__file__).parent / "analysis_output"
        # Find the latest top_k_with_iptc file
        export_files = list(analysis_dir.glob("top_k_with_iptc_*.csv"))
        if not export_files:
            # Check results dir as well (legacy)
            export_files = list(results_dir.glob("top_k_with_iptc_*.csv"))
        
        if export_files:
            latest_file = max(export_files, key=lambda f: f.stat().st_mtime)
            total_docs = pd.read_csv(latest_file)
            print(f" Loaded total docs from analysis output: {latest_file.name}")
        else:
            raise FileNotFoundError("No top_k_with_iptc file found")
    except (FileNotFoundError, PermissionError):
        # Fallback: compute from monthly data
        try:
            monthly = pd.read_csv(data_dir / "gdelt_monthly_docs_per_theme_country_2022_2024.csv")
            total_docs = (
                monthly
                .groupby(['country', 'theme_code'])['n_docs']
                .sum()
                .reset_index()
                .rename(columns={'n_docs': 'total_docs'})
            )
            print(" Computed total docs from monthly data")
        except FileNotFoundError:
            print(" Cannot find total docs data")
            return

    # Merge with IPTC mapping
    docs_with_iptc = total_docs.merge(
        iptc_mapping,
        on='theme_code',
        how='inner'
    )

    # Filter out metadata themes (same as analysis.py)
    meta_prefixes = ("TAX_", "WB_", "USPEC_", "UNGP_", "SOC_", "SLFID_")
    meta_exact = {"TAX", "WB", "USPEC", "UNGP", "SOC", "SLFID"}

    def is_metadata_theme(theme_code: str) -> bool:
        if theme_code in meta_exact:
            return True
        for prefix in meta_prefixes:
            if theme_code.startswith(prefix):
                return True
        return False

    docs_with_iptc = docs_with_iptc[~docs_with_iptc['theme_code'].apply(is_metadata_theme)]

    # Aggregate by IPTC category (sum across all countries)
    iptc_totals = (
        docs_with_iptc
        .groupby(['iptc_id', 'iptc_category', 'theme_code'])
        .agg(total_docs=('total_docs', 'sum'))
        .reset_index()
    )

    # Get unique IPTC categories
    iptc_categories = iptc_totals[['iptc_id', 'iptc_category']].drop_duplicates()

    # Create output directory
    output_dir = results_dir / "latex_treemaps"
    output_dir.mkdir(exist_ok=True)

    print(f" Generating treemaps for {len(iptc_categories)} IPTC categories...")

    # Generate treemap for each IPTC category
    for _, iptc_cat in iptc_categories.iterrows():
        iptc_id = iptc_cat['iptc_id']
        iptc_label = iptc_cat['iptc_category']

        # Get themes for this IPTC category
        category_themes = iptc_totals[iptc_totals['iptc_id'] == iptc_id].copy()

        if len(category_themes) == 0:
            print(f" No themes found for IPTC {iptc_id} ({iptc_label})")
            continue

        # Sort by total_docs descending
        category_themes = category_themes.sort_values('total_docs', ascending=False)

        # Prepare data for treemap
        treemap_data = category_themes[['theme_code', 'total_docs']].rename(columns={
            'theme_code': 'label',
            'total_docs': 'value'
        })

        # Generate filename
        safe_label = iptc_label.replace(' ', '_').replace(',', '').replace(' and ', '_')
        filename = f"treemap_iptc_{iptc_id}_{safe_label}.tex"
        out_path = output_dir / filename

        # Generate TikZ treemap
        export_treemap_tikz(
            df=treemap_data,
            label_col='label',
            value_col='value',
            out_path=out_path,
            width=120,
            height=80
        )

        print(f"   • {iptc_label}: {len(category_themes)} themes, {category_themes['total_docs'].sum():,} total docs")

    print(f"\n All IPTC treemaps generated in {output_dir}")


if __name__ == "__main__":
    generate_iptc_treemaps()