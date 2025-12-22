#!/usr/bin/env python3
"""
GDELT Theme Analysis for Country News Sentiment Study
======================================================

This script performs comprehensive GDELT theme analysis for 6 countries:
- CE (Central Europe)
- HO (Honduras)  
- HR (Croatia)
- KG (Kyrgyzstan)
- LO (Laos)
- SA (Saudi Arabia)

Data Sources:
- bquxjob_645c6baa_19b43fe1bcd.csv: Total docs per country-theme (2022-2024)
- bquxjob_4750d984_19b43fefa20.csv: Monthly quality metrics
- bquxjob_5c135702_19b43f3f270.csv: Monthly detail data

Author: CMPE490 Project
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

# Metadata prefixes - These are NOT news categories, just metadata tags
META_PREFIXES = ("TAX_", "WB_", "USPEC_", "UNGP_", "SOC_", "SLFID_")
META_EXACT = {"TAX", "WB", "USPEC", "UNGP", "SOC", "SLFID"}

# Usable news categories
USABLE_CATEGORIES = {
    'ECON', 'EPU_ECONOMY', 'EPU_POLICY', 'EPU',
    'GENERAL_GOVERNMENT', 'GENERAL', 'GOV',
    'CRISISLEX', 'ARMEDCONFLICT', 'TERROR', 'SECURITY', 'MILITARY',
    'HEALTH', 'GENERAL_HEALTH', 'MEDICAL',
    'EDUCATION',
    'TOURISM', 'AGRICULTURE',
    'LEADER', 'ELECTION', 'DEMOCRACY', 'LEGISLATION',
    'ENV', 'NATURAL', 'DISASTER',
    'MEDIA', 'SCIENCE', 'RELIGION'
}

# Category groupings for final analysis
CATEGORY_GROUPS = {
    'Economy': ['ECON', 'EPU_ECONOMY', 'EPU', 'EPU_POLICY'],
    'Government': ['GENERAL_GOVERNMENT', 'GENERAL', 'GOV', 'LEADER', 'ELECTION'],
    'Security': ['CRISISLEX', 'ARMEDCONFLICT', 'TERROR', 'SECURITY', 'MILITARY'],
    'Health': ['HEALTH', 'GENERAL_HEALTH', 'MEDICAL'],
    'Education': ['EDUCATION'],
    'Tourism': ['TOURISM']
}

# Countries (FIPS codes used by GDELT)
# CE = Sri Lanka (Ceylon), HO = Honduras, HR = Croatia, KG = Kyrgyzstan, LO = Slovakia, SA = Saudi Arabia
COUNTRIES = ['CE', 'HO', 'HR', 'KG', 'LO', 'SA']
COUNTRY_NAMES = {
    'CE': 'Sri Lanka',
    'HO': 'Honduras',
    'HR': 'Croatia',
    'KG': 'Kyrgyzstan',
    'LO': 'Slovakia',
    'SA': 'Saudi Arabia'
}

# Top-K themes per country
TOP_K = 15

# Thresholds for monthly sentiment analysis
MONTHLY_THRESHOLD = 0.7     # >= 70% months OK -> use monthly
QUARTERLY_THRESHOLD = 0.4  # >= 40% months OK -> use quarterly
MIN_DOCS_PER_MONTH = 50    # Minimum docs per month to be considered "OK"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_metadata_theme(theme_code: str) -> bool:
    """Check if a theme code is a metadata tag (not a news category)."""
    if theme_code in META_EXACT:
        return True
    for prefix in META_PREFIXES:
        if theme_code.startswith(prefix):
            return True
    return False


def get_theme_type(theme_code: str) -> str:
    """Classify theme as 'category', 'metadata', or 'other'."""
    if is_metadata_theme(theme_code):
        return 'metadata'
    if theme_code in USABLE_CATEGORIES:
        return 'category'
    return 'other'


def load_data(data_dir: Path = None) -> tuple:
    """
    Load all CSV data files.
    
    Returns:
        tuple: (total_docs_df, quality_df, monthly_df)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    
    # Total docs per country-theme
    total_docs = pd.read_csv(data_dir / "bquxjob_645c6baa_19b43fe1bcd.csv")
    
    # Monthly quality metrics
    quality = pd.read_csv(data_dir / "bquxjob_4750d984_19b43fefa20.csv")
    
    # Monthly detail (optional - might be large)
    try:
        monthly = pd.read_csv(data_dir / "bquxjob_5c135702_19b43f3f270.csv")
    except FileNotFoundError:
        monthly = None
        print("⚠️ Monthly detail file not found (optional)")
    
    return total_docs, quality, monthly


# ============================================================================
# STEP A: FILTER METADATA & SELECT TOP-K THEMES
# ============================================================================

def filter_metadata_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove metadata themes (TAX, WB, USPEC, etc.) from the dataframe.
    These are not news categories, just GDELT classification metadata.
    """
    mask = ~df['theme_code'].apply(is_metadata_theme)
    filtered = df[mask].copy()
    
    removed_count = len(df) - len(filtered)
    print(f"📊 Filtered out {removed_count} metadata theme rows")
    
    return filtered


def select_top_k_themes_per_country(df: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """
    For each country, select the top-K themes by total_docs.
    This identifies the dominant news topics per country.
    """
    top_k = (
        df
        .sort_values(['country', 'total_docs'], ascending=[True, False])
        .groupby('country')
        .head(k)
    )
    
    print(f"📊 Selected top {k} themes for each country")
    for country in COUNTRIES:
        country_themes = top_k[top_k['country'] == country]['theme_code'].tolist()
        print(f"   {country}: {', '.join(country_themes[:5])}...")
    
    return top_k


# ============================================================================
# STEP B: FIND COMMON THEMES ACROSS ALL COUNTRIES
# ============================================================================

def find_common_themes(top_k_df: pd.DataFrame) -> set:
    """
    Find themes that appear in the top-K list for ALL countries.
    This gives us a consistent set of categories to analyze.
    """
    themes_per_country = (
        top_k_df
        .groupby('country')['theme_code']
        .apply(set)
    )
    
    # Intersection of all sets
    if len(themes_per_country) == 0:
        return set()
    
    common_themes = set.intersection(*themes_per_country.values)
    
    print(f"\n✅ Found {len(common_themes)} common themes across all {len(COUNTRIES)} countries:")
    for theme in sorted(common_themes):
        total = top_k_df[top_k_df['theme_code'] == theme]['total_docs'].sum()
        print(f"   • {theme} ({total:,} total docs)")
    
    return common_themes


# ============================================================================
# STEP C: COVERAGE ANALYSIS (Monthly Quality)
# ============================================================================

def analyze_coverage(quality_df: pd.DataFrame, common_themes: set) -> pd.DataFrame:
    """
    For each country-theme pair, analyze monthly coverage quality.
    
    Decision Rules:
    - ratio >= 0.7: Use for monthly sentiment analysis
    - 0.4 <= ratio < 0.7: Use quarterly aggregation
    - ratio < 0.4: Exclude this category for this country
    """
    # Calculate ratio
    quality_df = quality_df.copy()
    quality_df['ratio_ok'] = (
        quality_df['months_ok'] / quality_df['months_total']
    ).fillna(0)
    
    # Filter to common themes only
    usable = quality_df[quality_df['theme_code'].isin(common_themes)].copy()
    
    # Add decision column
    def make_decision(ratio):
        if ratio >= MONTHLY_THRESHOLD:
            return 'monthly'
        elif ratio >= QUARTERLY_THRESHOLD:
            return 'quarterly'
        else:
            return 'exclude'
    
    usable['decision'] = usable['ratio_ok'].apply(make_decision)
    
    print(f"\n📊 Coverage Analysis Summary:")
    for decision in ['monthly', 'quarterly', 'exclude']:
        count = (usable['decision'] == decision).sum()
        print(f"   • {decision.upper()}: {count} country-theme pairs")
    
    return usable


# ============================================================================
# STEP D: GENERATE DECISION MATRIX (GOLDEN TABLE)
# ============================================================================

def generate_decision_matrix(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a country × theme decision matrix.
    
    This is the "golden table" that shows which categories 
    can be used for each country and how (monthly/quarterly/exclude).
    """
    # Pivot to wide format
    matrix = coverage_df.pivot(
        index='country',
        columns='theme_code',
        values='decision'
    ).fillna('exclude')
    
    # Reorder columns by overall usability
    theme_scores = coverage_df.groupby('theme_code')['ratio_ok'].mean().sort_values(ascending=False)
    ordered_themes = [t for t in theme_scores.index if t in matrix.columns]
    matrix = matrix[ordered_themes]
    
    return matrix


def generate_numeric_matrix(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numeric matrix: 1 = monthly usable, 0 = not usable.
    """
    usable = coverage_df[coverage_df['ratio_ok'] >= MONTHLY_THRESHOLD]
    
    matrix = (
        usable
        .pivot(index='country', columns='theme_code', values='ratio_ok')
        .notna()
        .astype(int)
    )
    
    return matrix


# ============================================================================
# STEP E: CATEGORY GROUP ANALYSIS
# ============================================================================

def analyze_category_groups(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each category group (Economy, Health, etc.), find the best theme
    per country based on coverage ratio.
    """
    results = []
    
    for group_name, themes in CATEGORY_GROUPS.items():
        group_data = coverage_df[coverage_df['theme_code'].isin(themes)]
        
        if group_data.empty:
            continue
            
        # For each country, find the best theme in this group
        for country in COUNTRIES:
            country_group = group_data[group_data['country'] == country]
            
            if country_group.empty:
                results.append({
                    'country': country,
                    'category_group': group_name,
                    'best_theme': None,
                    'ratio': 0,
                    'decision': 'exclude'
                })
            else:
                best = country_group.loc[country_group['ratio_ok'].idxmax()]
                results.append({
                    'country': country,
                    'category_group': group_name,
                    'best_theme': best['theme_code'],
                    'ratio': best['ratio_ok'],
                    'decision': best['decision'] if 'decision' in best else 'unknown',
                    'months_ok': best.get('months_ok', 0),
                    'months_total': best.get('months_total', 0),
                    'median_docs': best.get('median_docs', 0)
                })
    
    return pd.DataFrame(results)


def generate_group_matrix(group_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simplified country × category_group matrix with symbols.
    ✔ = monthly, ⚠ = quarterly, ✖ = exclude
    """
    def to_symbol(decision):
        if decision == 'monthly':
            return '✔'
        elif decision == 'quarterly':
            return '⚠'
        else:
            return '✖'
    
    matrix = group_analysis.pivot(
        index='country',
        columns='category_group',
        values='decision'
    ).fillna('exclude')
    
    # Apply symbols
    symbol_matrix = matrix.applymap(to_symbol)
    
    # Reorder columns
    group_order = list(CATEGORY_GROUPS.keys())
    ordered_cols = [c for c in group_order if c in symbol_matrix.columns]
    symbol_matrix = symbol_matrix[ordered_cols]
    
    return symbol_matrix


# ============================================================================
# STEP F: QUARTERLY AGGREGATION (Optional)
# ============================================================================

def add_quarter_column(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarter column for quarterly aggregation.
    Format: YYYY-QX (e.g., 2022-Q1)
    """
    if monthly_df is None or 'ym' not in monthly_df.columns:
        return None
    
    df = monthly_df.copy()
    
    # Parse year-month
    df['year'] = df['ym'].str[:4]
    df['month'] = df['ym'].str[-2:].astype(int)
    
    # Calculate quarter
    df['quarter_num'] = ((df['month'] - 1) // 3 + 1)
    df['quarter'] = df['year'] + '-Q' + df['quarter_num'].astype(str)
    
    return df


def aggregate_quarterly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly data to quarterly.
    """
    df = add_quarter_column(monthly_df)
    if df is None:
        return None
    
    quarterly = (
        df
        .groupby(['quarter', 'country', 'theme_code'])
        .agg({
            'n_docs': 'sum'
        })
        .reset_index()
    )
    
    return quarterly


# ============================================================================
# REPORTING
# ============================================================================

def generate_report_sentences() -> list:
    """
    Generate ready-to-use sentences for the academic report.
    """
    return [
        "We identified the most frequent GDELT theme families for each country and retained those that consistently appeared across all six countries and corresponded to meaningful news topics.",
        f"Metadata-only tags (TAX, WB, USPEC, etc.) were filtered out as they represent GDELT's internal classification rather than actual news content.",
        f"To ensure robust sentiment aggregation, we required at least {MIN_DOCS_PER_MONTH} distinct news records per country–theme–month.",
        f"Categories with monthly coverage ≥{int(MONTHLY_THRESHOLD*100)}% were used for monthly sentiment analysis; those with {int(QUARTERLY_THRESHOLD*100)}-{int(MONTHLY_THRESHOLD*100)}% coverage were aggregated quarterly; below {int(QUARTERLY_THRESHOLD*100)}% were excluded.",
        "The final category selection provides adequate temporal granularity for cross-country sentiment comparison while maintaining statistical validity."
    ]


def print_summary(decision_matrix: pd.DataFrame, group_matrix: pd.DataFrame):
    """Print a nicely formatted summary."""
    print("\n" + "="*70)
    print("📊 GDELT THEME ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n📋 Individual Theme Decision Matrix:")
    print(decision_matrix.to_string())
    
    print("\n\n📋 Category Group Matrix (✔=monthly, ⚠=quarterly, ✖=exclude):")
    print(group_matrix.to_string())
    
    print("\n\n📝 Report Sentences:")
    for i, sentence in enumerate(generate_report_sentences(), 1):
        print(f"   {i}. {sentence}")
    
    print("\n" + "="*70)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results(
    decision_matrix: pd.DataFrame,
    group_matrix: pd.DataFrame,
    group_analysis: pd.DataFrame,
    coverage_df: pd.DataFrame,
    output_dir: Path = None
):
    """Export all results to CSV files."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "analysis_output"
    
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Decision matrix
    decision_matrix.to_csv(output_dir / f"decision_matrix_{timestamp}.csv")
    
    # Group matrix
    group_matrix.to_csv(output_dir / f"group_matrix_{timestamp}.csv")
    
    # Detailed group analysis
    group_analysis.to_csv(output_dir / f"group_analysis_{timestamp}.csv", index=False)
    
    # Full coverage data
    coverage_df.to_csv(output_dir / f"coverage_analysis_{timestamp}.csv", index=False)
    
    # Report sentences
    with open(output_dir / f"report_sentences_{timestamp}.txt", 'w', encoding='utf-8') as f:
        for sentence in generate_report_sentences():
            f.write(sentence + "\n\n")
    
    print(f"\n✅ Results exported to {output_dir}")


def export_to_json(
    decision_matrix: pd.DataFrame,
    group_matrix: pd.DataFrame,
    group_analysis: pd.DataFrame,
    output_file: Path = None
):
    """Export results as JSON for web consumption."""
    if output_file is None:
        output_file = Path(__file__).parent / "gdelt_analysis_results.json"
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'countries': COUNTRIES,
        'category_groups': CATEGORY_GROUPS,
        'thresholds': {
            'monthly': MONTHLY_THRESHOLD,
            'quarterly': QUARTERLY_THRESHOLD,
            'min_docs': MIN_DOCS_PER_MONTH
        },
        'decision_matrix': decision_matrix.to_dict(),
        'group_matrix': group_matrix.to_dict(),
        'group_analysis': group_analysis.to_dict(orient='records'),
        'report_sentences': generate_report_sentences()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON results exported to {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_analysis(data_dir: Path = None, export: bool = True) -> dict:
    """
    Run the complete GDELT theme analysis pipeline.
    
    Returns:
        dict: Analysis results including matrices and statistics
    """
    print("🚀 Starting GDELT Theme Analysis...")
    print("="*70)
    
    # Step 1: Load data
    print("\n📥 Loading data...")
    total_docs, quality, monthly = load_data(data_dir)
    print(f"   • Total docs: {len(total_docs)} rows")
    print(f"   • Quality: {len(quality)} rows")
    print(f"   • Monthly: {len(monthly) if monthly is not None else 'N/A'} rows")
    
    # Step 2: Filter metadata themes
    print("\n🔍 Step A: Filtering metadata themes...")
    filtered_docs = filter_metadata_themes(total_docs)
    
    # Step 3: Select top-K themes per country
    print(f"\n📊 Step B: Selecting top-{TOP_K} themes per country...")
    top_k = select_top_k_themes_per_country(filtered_docs, TOP_K)
    
    # Step 4: Find common themes
    print("\n🔗 Step C: Finding common themes across all countries...")
    common_themes = find_common_themes(top_k)
    
    # Step 5: Coverage analysis
    print("\n📈 Step D: Analyzing monthly coverage quality...")
    coverage = analyze_coverage(quality, common_themes)
    
    # Step 6: Generate decision matrix
    print("\n📋 Step E: Generating decision matrices...")
    decision_matrix = generate_decision_matrix(coverage)
    numeric_matrix = generate_numeric_matrix(coverage)
    
    # Step 7: Category group analysis
    print("\n🏷️ Step F: Analyzing category groups...")
    group_analysis = analyze_category_groups(coverage)
    group_matrix = generate_group_matrix(group_analysis)
    
    # Print summary
    print_summary(decision_matrix, group_matrix)
    
    # Export if requested
    if export:
        export_results(decision_matrix, group_matrix, group_analysis, coverage)
        export_to_json(decision_matrix, group_matrix, group_analysis)
    
    return {
        'common_themes': common_themes,
        'coverage': coverage,
        'decision_matrix': decision_matrix,
        'numeric_matrix': numeric_matrix,
        'group_analysis': group_analysis,
        'group_matrix': group_matrix
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GDELT Theme Analysis for Country News Study'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=None,
        help='Directory containing CSV files (default: script directory)'
    )
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip exporting results to files'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=TOP_K,
        help=f'Number of top themes per country (default: {TOP_K})'
    )
    parser.add_argument(
        '--monthly-threshold', '-m',
        type=float,
        default=MONTHLY_THRESHOLD,
        help=f'Monthly usability threshold (default: {MONTHLY_THRESHOLD})'
    )
    
    args = parser.parse_args()
    
    # Update globals if specified
    if args.top_k != TOP_K:
        TOP_K = args.top_k
    if args.monthly_threshold != MONTHLY_THRESHOLD:
        MONTHLY_THRESHOLD = args.monthly_threshold
    
    # Run analysis
    results = run_analysis(
        data_dir=args.data_dir,
        export=not args.no_export
    )
    
    print("\n✅ Analysis complete!")
