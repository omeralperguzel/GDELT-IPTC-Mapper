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

Data Sources (Unified Approach):
- gdelt_monthly_docs_per_theme_country_2022_2024.csv: Monthly document counts by theme and country (primary source)
- Total docs and quality metrics are computed from monthly detail to ensure consistency
- Optional: gdelt_theme_trend_analysis_2022_2024.csv (trend analysis for enhanced decisions)
- Original BigQuery exports: gdelt_top15_themes_by_country_2022_2024.csv (top 15 per country)

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


def compute_total_from_monthly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total_docs per country-theme from monthly detail data.
    This replaces the separate total_docs CSV.
    """
    if monthly_df is None:
        return None
    
    total_docs = (
        monthly_df
        .groupby(['country', 'theme_code'])['n_docs']
        .sum()
        .reset_index()
        .rename(columns={'n_docs': 'total_docs'})
    )
    
    print(f"✅ Computed total_docs from monthly data: {len(total_docs)} rows")
    return total_docs


def compute_quality_from_monthly(monthly_df: pd.DataFrame, min_docs: int = MIN_DOCS_PER_MONTH) -> pd.DataFrame:
    """
    Compute quality metrics from monthly detail data.
    This replaces the separate quality CSV.
    """
    if monthly_df is None:
        return None
    
    quality = (
        monthly_df
        .groupby(['country', 'theme_code'])
        .agg(
            months_total=('ym', 'count'),
            months_ok=('n_docs', lambda x: (x >= min_docs).sum()),
            min_docs=('n_docs', 'min'),
            median_docs=('n_docs', lambda x: x.median())
        )
        .reset_index()
    )
    
    print(f"✅ Computed quality metrics from monthly data: {len(quality)} rows")
    return quality


def load_data(data_dir: Path = None) -> tuple:
    """
    Load all CSV data files.
    
    Returns:
        tuple: (total_docs_df, quality_df, monthly_df, trend_df, iptc_mapping_df)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    
    # Monthly detail (primary data source)
    try:
        monthly = pd.read_csv(data_dir / "gdelt_monthly_docs_per_theme_country_2022_2024.csv")
        print("✅ Monthly detail loaded as primary data source")
    except FileNotFoundError:
        monthly = None
        print("❌ Monthly detail file not found - cannot proceed")
        return None, None, None, None, None
    
    # Compute total_docs from monthly detail
    total_docs = compute_total_from_monthly(monthly)
    
    # Compute quality metrics from monthly detail
    quality = compute_quality_from_monthly(monthly)
    
    # Load trend analysis (optional enhancement)
    try:
        trend = pd.read_csv(data_dir / "gdelt_theme_trend_analysis_2022_2024.csv")
        print("✅ Trend analysis loaded")
    except FileNotFoundError:
        trend = None
        print("⚠️ Trend analysis file not found (optional)")
    
    # Load IPTC mapping (for category integration)
    try:
        with open(data_dir / "gdelt_iptc_mapping_v2.json", 'r', encoding='utf-8') as f:
            iptc_data = json.load(f)
        iptc_mapping = pd.DataFrame(iptc_data['themes'])[['theme_code', 'iptc_final_id', 'iptc_final_label']]
        iptc_mapping = iptc_mapping.rename(columns={'iptc_final_id': 'iptc_id', 'iptc_final_label': 'iptc_category'})
        print("✅ IPTC mapping loaded")
    except (FileNotFoundError, KeyError):
        iptc_mapping = None
        print("⚠️ IPTC mapping file not found (optional)")
    
    return total_docs, quality, monthly, trend, iptc_mapping


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

def analyze_coverage(quality_df: pd.DataFrame, common_themes: set, trend_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each country-theme pair, analyze monthly coverage quality.
    Enhanced with trend analysis for better decision making.
    
    Decision Rules:
    - ratio >= 0.7 AND trend stable/growing: Use for monthly sentiment analysis
    - ratio >= 0.4 OR (ratio >= 0.6 AND growing): Use quarterly aggregation
    - ratio < 0.4 OR declining: Exclude this category for this country
    """
    # Calculate ratio
    quality_df = quality_df.copy()
    quality_df['ratio_ok'] = (
        quality_df['months_ok'] / quality_df['months_total']
    ).fillna(0)
    
    # Filter to common themes only
    usable = quality_df[quality_df['theme_code'].isin(common_themes)].copy()
    
    # Add trend information if available
    if trend_df is not None:
        usable = usable.merge(
            trend_df[['country', 'theme_code', 'trend_category', 'std_docs']],
            on=['country', 'theme_code'],
            how='left'
        )
        usable['trend_category'] = usable['trend_category'].fillna('Unknown')
        usable['std_docs'] = usable['std_docs'].fillna(usable['std_docs'].mean())
    else:
        usable['trend_category'] = 'Unknown'
        usable['std_docs'] = usable['median_docs']  # Fallback
    
    # Enhanced decision function
    def make_decision_enhanced(row):
        ratio = row['ratio_ok']
        trend = row['trend_category']
        std_ratio = row['std_docs'] / row['median_docs'] if row['median_docs'] > 0 else 1
        
        # Prefer stable or growing themes with good coverage
        if ratio >= 0.7 and trend in ['Stabil', 'Artan']:
            return 'monthly'
        elif ratio >= 0.6 and trend == 'Artan':
            return 'monthly'
        elif ratio >= 0.4 and std_ratio < 0.5:  # Low variance
            return 'quarterly'
        elif ratio >= 0.5:
            return 'quarterly'
        else:
            return 'exclude'
    
    usable['decision'] = usable.apply(make_decision_enhanced, axis=1)
    
    print(f"\n📊 Enhanced Coverage Analysis Summary:")
    for decision in ['monthly', 'quarterly', 'exclude']:
        count = (usable['decision'] == decision).sum()
        print(f"   • {decision.upper()}: {count} country-theme pairs")
    
    if trend_df is not None:
        trend_counts = usable['trend_category'].value_counts()
        print(f"   • Trend Distribution: {dict(trend_counts)}")
    
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
    symbol_matrix = matrix.apply(lambda x: x.map(to_symbol))
    
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
    Updated to reflect unified data source approach.
    """
    return [
        "We identified the most frequent GDELT theme families for each country and retained those that consistently appeared across all six countries and corresponded to meaningful news topics.",
        f"Metadata-only tags (TAX, WB, USPEC, etc.) were filtered out as they represent GDELT's internal classification rather than actual news content.",
        f"To ensure robust sentiment aggregation, we required at least {MIN_DOCS_PER_MONTH} distinct news records per country–theme–month.",
        f"Categories with monthly coverage ≥{int(MONTHLY_THRESHOLD*100)}% were used for monthly sentiment analysis; those with {int(QUARTERLY_THRESHOLD*100)}-{int(MONTHLY_THRESHOLD*100)}% coverage were aggregated quarterly; below {int(QUARTERLY_THRESHOLD*100)}% were excluded.",
        "All analyses were performed using unified monthly detail data to ensure consistency across total document counts, quality metrics, and coverage calculations."
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
    results: dict,
    output_file: Path = None
):
    """Export results as JSON for web consumption."""
    if output_file is None:
        output_file = Path(__file__).parent / "gdelt_analysis_results.json"
    
    json_results = {
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
    
    # Yeni IPTC ve trend verilerini ekle
    if results.get('category_stats') is not None:
        json_results['category_stats'] = results['category_stats'].to_dict(orient='records')
    if results.get('country_cat_coverage') is not None:
        json_results['country_cat_coverage'] = results['country_cat_coverage'].to_dict(orient='records')
    if results.get('trend_with_coverage') is not None:
        json_results['trend_with_coverage'] = results['trend_with_coverage'].to_dict(orient='records')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
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
    total_docs, quality, monthly, trend, iptc_mapping = load_data(data_dir)
    
    if monthly is None:
        print("❌ Cannot proceed without monthly detail data")
        return {}
    
    print(f"   • Monthly detail: {len(monthly)} rows")
    print(f"   • Computed total_docs: {len(total_docs)} rows")
    print(f"   • Computed quality: {len(quality)} rows")
    print(f"   • Trend analysis: {len(trend) if trend is not None else 'N/A'} rows")
    print(f"   • IPTC mapping: {len(iptc_mapping) if iptc_mapping is not None else 'N/A'} rows")
    
    # ============================================================================
    # STEP A: TEMA SEÇİMİ VE IPTC ÖZETLERİ
    # ============================================================================
    
    print("\n🔍 Adım A: Tema seçimi ve IPTC özetleri...")
    
    # Filter metadata themes
    filtered_docs = filter_metadata_themes(total_docs)
    
    # Select top-K themes per country
    top_k = select_top_k_themes_per_country(filtered_docs, TOP_K)
    
    # Find common themes
    common_themes = find_common_themes(top_k)
    
    # IPTC entegrasyonu: top_k_df'yi IPTC mapping ile birleştir
    top_k_with_iptc = None
    common_with_iptc = None
    category_stats = None
    
    if iptc_mapping is not None:
        top_k_with_iptc = top_k.merge(
            iptc_mapping[["theme_code", "iptc_id", "iptc_category"]],
            on="theme_code",
            how="left"
        )
        
        common_with_iptc = top_k_with_iptc[
            top_k_with_iptc["theme_code"].isin(common_themes)
        ]
        
        # Kategori bazında özet istatistikleri hesapla
        category_stats = (
            common_with_iptc
            .groupby("iptc_id")
            .agg(
                theme_count=("theme_code", "nunique"),
                doc_count=("total_docs", "sum")
            )
            .reset_index()
        )
        
        print(f"✅ IPTC entegrasyonu tamamlandı: {len(category_stats)} kategori bulundu")
    
    # ============================================================================
    # STEP B: ÜLKE × IPTC KULLANILABILIRLIK MATRISI
    # ============================================================================
    
    print("\n📊 Adım B: Ülke × IPTC kullanılabilirlik matrisi...")
    
    # Coverage analysis
    coverage = analyze_coverage(quality, common_themes, trend)
    
    # IPTC'ye göre aggregate et
    country_cat_coverage = None
    
    if iptc_mapping is not None:
        coverage_with_iptc = coverage.merge(
            iptc_mapping[["theme_code", "iptc_id"]],
            on="theme_code",
            how="left"
        )
        
        country_cat_coverage = (
            coverage_with_iptc
            .groupby(["country", "iptc_id"])
            .agg(
                months_total=("months_total", "sum"),
                months_ok=("months_ok", "sum")
            )
            .reset_index()
        )
        
        country_cat_coverage["ratio_ok"] = (
            country_cat_coverage["months_ok"] /
            country_cat_coverage["months_total"]
        )
        
        # Karar ver
        def decide_level(r):
            if r["ratio_ok"] >= MONTHLY_THRESHOLD:
                return "monthly"
            elif r["ratio_ok"] >= QUARTERLY_THRESHOLD:
                return "quarterly"
            else:
                return "exclude"
        
        country_cat_coverage["coverage_level"] = country_cat_coverage.apply(decide_level, axis=1)
        
        print(f"✅ Ülke × IPTC matrisi oluşturuldu: {len(country_cat_coverage)} satır")
    
    # ============================================================================
    # STEP C: ANALIZ İÇİN KATEGORI/ÜLKE SEÇİMİ VE TREND
    # ============================================================================
    
    print("\n📈 Adım C: Analiz için kategori/ülke seçimi ve trend...")
    
    # Trend + coverage birleşimi
    trend_with_coverage = None
    
    if trend is not None:
        trend_with_coverage = trend.merge(
            coverage[["country", "theme_code", "decision"]],
            on=["country", "theme_code"],
            how="left"
        )
        
        trend_with_coverage = trend_with_coverage[
            trend_with_coverage["decision"].isin(["monthly", "quarterly"])
        ]
        
        print(f"✅ Trend + coverage birleşimi: {len(trend_with_coverage)} satır")
    
    # Eski adımlar (karar matrisi vb.)
    print("\n📋 Ek analizler...")
    decision_matrix = generate_decision_matrix(coverage)
    numeric_matrix = generate_numeric_matrix(coverage)
    group_analysis = analyze_category_groups(coverage)
    group_matrix = generate_group_matrix(group_analysis)
    
    # Print summary
    print_summary(decision_matrix, group_matrix)
    
    # Export if requested
    if export:
        export_results(decision_matrix, group_matrix, group_analysis, coverage)
        export_to_json(decision_matrix, group_matrix, group_analysis, {
            'category_stats': category_stats,
            'country_cat_coverage': country_cat_coverage,
            'trend_with_coverage': trend_with_coverage
        })
        
        # Yeni export'lar
        if top_k_with_iptc is not None:
            top_k_with_iptc.to_csv(Path(__file__).parent / "analysis_output" / f"top_k_with_iptc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        if country_cat_coverage is not None:
            country_cat_coverage.to_csv(Path(__file__).parent / "analysis_output" / f"country_cat_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        if trend_with_coverage is not None:
            trend_with_coverage.to_csv(Path(__file__).parent / "analysis_output" / f"trend_with_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    return {
        'common_themes': common_themes,
        'coverage': coverage,
        'decision_matrix': decision_matrix,
        'numeric_matrix': numeric_matrix,
        'group_analysis': group_analysis,
        'group_matrix': group_matrix,
        'top_k_with_iptc': top_k_with_iptc,
        'common_with_iptc': common_with_iptc,
        'category_stats': category_stats,
        'country_cat_coverage': country_cat_coverage,
        'trend_with_coverage': trend_with_coverage
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
