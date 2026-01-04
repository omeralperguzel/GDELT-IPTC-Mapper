#!/usr/bin/env python3
"""
GDELT Theme Semantic Clustering (RTX 3050 optimized)
=====================================================

Pipeline:
1. Load theme list from BigQuery CSV
2. Merge with Vargo issue classification (external prior)
3. Build text representations (theme_code + issue + description)
4. Sentence-Transformer embedding (all-MiniLM-L6-v2) → 384-dim
5. PCA dimensionality reduction → 20-dim
6. Agglomerative clustering → 8 IPTC-aligned clusters
7. Label clusters based on dominant issue category
8. Export results for web visualization

References:
- Tarekegn (2024): Event clustering with LLM embeddings
- Kuzman & Ljubešić (2024): IPTC top-level news classification
- Vargo & Guo: GDELT theme-to-issue mapping
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try to import ML libraries - gracefully degrade if not available
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("  sentence-transformers not available - using fallback mode")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  scikit-learn not available")


# ============================================================================
# CONFIGURATION
# ============================================================================

# IPTC Media Topics top-level categories (17 categories for v3 compatibility)
IPTC_CATEGORIES = {
    'Arts': 'arts, culture, entertainment and media',
    'Crime': 'crime, law and justice',
    'Disaster': 'disaster, accident and emergency incident',
    'Economy': 'economy, business and finance',
    'Education': 'education',
    'Environment': 'environment',
    'Health': 'health',
    'Human Interest': 'human interest',
    'Labour': 'labour',
    'Lifestyle': 'lifestyle and leisure',
    'Politics': 'politics and government',
    'Religion': 'religion',
    'Science': 'science and technology',
    'Society': 'society',
    'Sport': 'sport',
    'Conflict': 'conflict, war and peace',
    'Weather': 'weather'
}

# Number of clusters to create (17 for IPTC categories)
N_CLUSTERS = 17

# Embedding model (light, efficient for RTX 3050)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# PCA target dimensions
PCA_COMPONENTS = 20


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_themes_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load unique themes from BigQuery total_docs CSV."""
    try:
        df = pd.read_csv(csv_path)
        # Get unique themes
        themes = df[['theme_code']].drop_duplicates().reset_index(drop=True)
        print(f" Loaded {len(themes)} unique themes from {csv_path.name}")
        return themes
    except FileNotFoundError:
        print(f" File not found: {csv_path}")
        return pd.DataFrame()


def load_vargo_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load Vargo theme-to-issue mapping."""
    try:
        df = pd.read_csv(mapping_path)
        print(f" Loaded Vargo mapping with {len(df)} theme-issue pairs")
        return df
    except FileNotFoundError:
        print(f"  Vargo mapping not found at {mapping_path}")
        return pd.DataFrame()


def build_theme_text_repr(theme_code: str, issue: str = None, description: str = None) -> str:
    """
    Build text representation for theme.
    Format: "THEME_CODE - ISSUE_CATEGORY - description"
    """
    parts = [theme_code]
    
    if pd.notna(issue) and issue:
        parts.append(str(issue).strip())
    
    if pd.notna(description) and description:
        parts.append(str(description).strip())
    
    return " - ".join(parts)


def generate_embeddings(texts: list) -> np.ndarray:
    """
    Generate sentence-transformer embeddings.
    Falls back to random embeddings if library not available.
    """
    if not HAS_TRANSFORMERS:
        print("  Using random embeddings (sentence-transformers not installed)")
        return np.random.randn(len(texts), 384).astype(np.float32)
    
    print(f" Generating embeddings for {len(texts)} themes...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f" Embeddings generated: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f" Embedding error: {e}")
        return np.random.randn(len(texts), 384).astype(np.float32)


def reduce_dimensions(embeddings: np.ndarray) -> tuple:
    """
    PCA dimensionality reduction: 384 → 20 dimensions (or min available if fewer samples).
    """
    if not HAS_SKLEARN:
        print("  Skipping PCA (scikit-learn not installed)")
        return embeddings, None
    
    # Ensure n_components doesn't exceed number of samples or features
    max_components = min(embeddings.shape[0], embeddings.shape[1])
    actual_components = min(PCA_COMPONENTS, max_components)
    
    print(f" Reducing dimensions: {embeddings.shape[1]} → {actual_components}")
    pca = PCA(n_components=actual_components, random_state=42)
    X_reduced = pca.fit_transform(embeddings)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f" Reduced to {actual_components} dimensions (explains {explained_var:.1%} variance)")
    
    return X_reduced, pca


def cluster_themes(X: np.ndarray, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """
    Agglomerative clustering: Ward linkage.
    """
    if not HAS_SKLEARN:
        print("  Skipping clustering (scikit-learn not installed)")
        return np.zeros(len(X), dtype=int)
    
    print(f" Clustering with Agglomerative (k={n_clusters})...")
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    labels = clusterer.fit_predict(X)
    
    for i in range(n_clusters):
        count = (labels == i).sum()
        print(f"   Cluster {i}: {count} themes")
    
    return labels


def label_clusters(df: pd.DataFrame) -> dict:
    """
    Label clusters based on dominant issue category.
    Map to IPTC Media Topics (17 categories for v3 compatibility).
    """
    cluster_labels = {}

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]

        # Get issue distribution
        if 'issue_category' in df.columns:
            issue_dist = cluster_data['issue_category'].value_counts()
            dominant_issue = issue_dist.index[0] if len(issue_dist) > 0 else "Unknown"
        else:
            dominant_issue = "Unknown"

        # Sample themes from cluster
        sample_themes = cluster_data['theme_code'].head(10).tolist()

        # Map to IPTC category (expanded for 17 categories)
        iptc_label = "Society"  # default
        if 'ECONOMY' in dominant_issue or 'FINANCE' in dominant_issue or 'BUSINESS' in dominant_issue:
            iptc_label = "Economy"
        elif 'POLITIC' in dominant_issue or 'GOVERN' in dominant_issue or 'DIPLOMACY' in dominant_issue:
            iptc_label = "Politics"
        elif 'CONFLICT' in dominant_issue or 'WAR' in dominant_issue or 'TERROR' in dominant_issue or 'MILITARY' in dominant_issue:
            iptc_label = "Conflict"
        elif 'HEALTH' in dominant_issue or 'MEDICAL' in dominant_issue or 'DISEASE' in dominant_issue:
            iptc_label = "Health"
        elif 'EDUC' in dominant_issue or 'SCHOOL' in dominant_issue or 'UNIVERSITY' in dominant_issue:
            iptc_label = "Education"
        elif 'ENV' in dominant_issue or 'CLIMATE' in dominant_issue or 'POLLUTION' in dominant_issue:
            iptc_label = "Environment"
        elif 'CRIME' in dominant_issue or 'LAW' in dominant_issue or 'JUSTICE' in dominant_issue:
            iptc_label = "Crime"
        elif 'DISASTER' in dominant_issue or 'ACCIDENT' in dominant_issue or 'EMERGENCY' in dominant_issue:
            iptc_label = "Disaster"
        elif 'ARTS' in dominant_issue or 'CULTURE' in dominant_issue or 'MEDIA' in dominant_issue or 'ENTERTAINMENT' in dominant_issue:
            iptc_label = "Arts"
        elif 'RELIGION' in dominant_issue or 'FAITH' in dominant_issue:
            iptc_label = "Religion"
        elif 'SCIENCE' in dominant_issue or 'TECH' in dominant_issue or 'RESEARCH' in dominant_issue:
            iptc_label = "Science"
        elif 'SPORT' in dominant_issue or 'ATHLET' in dominant_issue:
            iptc_label = "Sport"
        elif 'WEATHER' in dominant_issue or 'CLIMATE' in dominant_issue:
            iptc_label = "Weather"
        elif 'LABOUR' in dominant_issue or 'EMPLOYMENT' in dominant_issue or 'WORK' in dominant_issue:
            iptc_label = "Labour"
        elif 'HUMAN' in dominant_issue or 'INTEREST' in dominant_issue:
            iptc_label = "Human Interest"
        elif 'LIFESTYLE' in dominant_issue or 'LEISURE' in dominant_issue or 'TRAVEL' in dominant_issue:
            iptc_label = "Lifestyle"

        cluster_labels[int(cluster_id)] = {
            'iptc_label': iptc_label,
            'iptc_description': IPTC_CATEGORIES.get(iptc_label, iptc_label),
            'dominant_issue': dominant_issue,
            'sample_themes': sample_themes,
            'theme_count': len(cluster_data)
        }

    return cluster_labels


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_clustering_pipeline(data_dir: Path = None, output_dir: Path = None) -> dict:
    """
    Full clustering pipeline for GDELT themes.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    
    print(" GDELT Theme Semantic Clustering Pipeline")
    print("=" * 70)
    
    # Step 1: Load themes from BigQuery CSV
    print("\n Step 1: Loading themes from BigQuery export...")
    themes = load_themes_from_csv(data_dir / "bquxjob_645c6baa_19b43fe1bcd.csv")
    
    if themes.empty:
        print(" No themes loaded. Exiting.")
        return {}
    
    # Step 2: Load Vargo mapping (external prior)
    print("\n Step 2: Loading Vargo theme-to-issue mapping...")
    vargo = load_vargo_mapping(data_dir / "vargo_gdelt_themes_issues.csv")
    
    if not vargo.empty:
        themes = themes.merge(
            vargo,
            left_on='theme_code',
            right_on='theme_code',
            how='left'
        )
    
    # Step 3: Build text representations
    print("\n Step 3: Building text representations...")
    themes['text_repr'] = themes.apply(
        lambda row: build_theme_text_repr(
            row['theme_code'],
            row.get('issue_category'),
            row.get('description')
        ),
        axis=1
    )
    
    print("   Sample texts:")
    for i, text in enumerate(themes['text_repr'].head(3)):
        print(f"   {i+1}. {text[:80]}...")
    
    # Step 4: Generate embeddings
    print("\n Step 4: Generating sentence-transformer embeddings...")
    embeddings = generate_embeddings(themes['text_repr'].tolist())
    
    # Step 5: PCA dimensionality reduction
    print("\n Step 5: Reducing dimensions with PCA...")
    X_reduced, pca_model = reduce_dimensions(embeddings)
    
    # Step 6: Agglomerative clustering
    print("\n Step 6: Clustering themes...")
    labels = cluster_themes(X_reduced, n_clusters=N_CLUSTERS)
    themes['cluster'] = labels
    
    # Step 7: Label clusters
    print("\n  Step 7: Labeling clusters...")
    cluster_labels = label_clusters(themes)
    
    # Print summary
    print("\n" + "=" * 70)
    print(" CLUSTERING RESULTS SUMMARY")
    print("=" * 70)
    for cluster_id, info in sorted(cluster_labels.items()):
        print(f"\n Cluster {cluster_id}: {info['iptc_label']}")
        print(f"   IPTC: {info['iptc_description']}")
        print(f"   Dominant Issue: {info['dominant_issue']}")
        print(f"   Themes: {info['theme_count']}")
        print(f"   Examples: {', '.join(info['sample_themes'][:5])}")
    
    # Prepare results for export
    results = {
        'metadata': {
            'model': EMBEDDING_MODEL,
            'pca_components': PCA_COMPONENTS,
            'n_clusters': N_CLUSTERS,
            'total_themes': len(themes),
            'algorithm': 'Agglomerative Clustering (Ward linkage)'
        },
        'cluster_labels': cluster_labels,
        'themes': themes[[
            'theme_code', 'cluster', 'issue_category', 'text_repr'
        ]].to_dict(orient='records'),
        'explained_variance': float(pca_model.explained_variance_ratio_.sum()) if pca_model else 0.0
    }
    
    return results


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results(results: dict, output_dir: Path = None):
    """Export clustering results to JSON and CSV."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    output_dir.mkdir(exist_ok=True)
    
    # NaN değerlerini null ile değiştir
    def convert_nan(obj):
        if isinstance(obj, dict):
            return {k: convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan(v) for v in obj]
        elif isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
        return obj
    
    results_clean = convert_nan(results)
    
    # JSON export (for web)
    json_file = output_dir / "gdelt_theme_clusters.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    print(f" Exported: {json_file}")
    
    # CSV export (for reference)
    if 'themes' in results:
        csv_df = pd.DataFrame(results['themes'])
        csv_file = output_dir / "gdelt_themes_with_clusters.csv"
        csv_df.to_csv(csv_file, index=False)
        print(f" Exported: {csv_file}")
    
    return json_file, csv_file


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GDELT Theme Semantic Clustering'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=None,
        help='Directory containing BigQuery CSV files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Directory to save output files'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        default=True,
        help='Export results to JSON and CSV'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_clustering_pipeline(args.data_dir, args.output_dir)
    
    # Export
    if args.export and results:
        export_results(results, args.output_dir)
    
    print("\n Pipeline complete!")
