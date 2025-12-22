#!/usr/bin/env python3
"""
GDELT Theme → IPTC Media Topics Semantic Mapping

Bu script GDELT temalarını IPTC Media Topics taksonomisine eşler.
Yaklaşım:
1. IPTC üst düzey konularını (17 kategori) embedding uzayına yerleştir
2. GDELT temalarını aynı uzaya yerleştir  
3. Her GDELT temasını en yakın IPTC konusuna cosine similarity ile ata

Referanslar:
- Kuzman & Ljubešić (2024): IPTC top-level news classification
- Tarekegn (2024): GDELT event clustering with LLM embeddings
- IPTC NewsCodes: https://iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html

RTX 3050 optimized: all-MiniLM-L6-v2 (384-dim, ~80MB)
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[!] sentence-transformers not available - using fallback mode")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[!] scikit-learn not available")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sentence-Transformer model (RTX 3050 optimized)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Files
IPTC_FILE = "iptc_mediatopics.csv"
VARGO_FILE = "vargo_gdelt_themes_issues.csv"
GDELT_CSV_FILES = [
    'gdelt_top15_themes_by_country_2022_2024.csv',
    'gdelt_monthly_quality_metrics.csv',
    'gdelt_monthly_docs_per_theme_country_2022_2024.csv'
]

# Output
OUTPUT_JSON = "gdelt_iptc_mapping_v1.json"
OUTPUT_CSV = "gdelt_themes_iptc_v1.csv"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_iptc_topics(data_dir: Path) -> pd.DataFrame:
    """Load IPTC Media Topics taxonomy."""
    iptc_file = data_dir / IPTC_FILE
    
    if not iptc_file.exists():
        raise FileNotFoundError(f"IPTC file not found: {iptc_file}")
    
    iptc = pd.read_csv(iptc_file)
    
    # Create text representation for embedding
    iptc["text_repr"] = iptc["label"] + " - " + iptc["definition"]
    
    print(f"[OK] Loaded {len(iptc)} IPTC topics ({len(iptc[iptc['level']==1])} top-level)")
    return iptc


def load_gdelt_themes(data_dir: Path) -> pd.DataFrame:
    """Load unique GDELT themes from all BigQuery export files."""
    theme_codes = set()
    files_loaded = 0
    
    for csv_name in GDELT_CSV_FILES:
        csv_file = data_dir / csv_name
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                if 'theme_code' in df.columns:
                    theme_codes.update(df['theme_code'].dropna().unique())
                    files_loaded += 1
            except Exception as e:
                print(f"[!] Could not read {csv_name}: {e}")
    
    if not theme_codes:
        raise FileNotFoundError(f"No GDELT themes found in CSV files")
    
    # Convert to DataFrame
    themes = pd.DataFrame({'theme_code': sorted(theme_codes)})
    
    print(f"[OK] Loaded {len(themes)} unique GDELT themes from {files_loaded} CSV files")
    return themes


def load_vargo_mapping(data_dir: Path) -> pd.DataFrame:
    """Load Vargo theme-to-issue mapping for additional context."""
    vargo_file = data_dir / VARGO_FILE
    
    if not vargo_file.exists():
        print(f"[!] Vargo mapping not found: {vargo_file}")
        return pd.DataFrame()
    
    vargo = pd.read_csv(vargo_file)
    print(f"[OK] Loaded Vargo mapping with {len(vargo)} theme-issue pairs")
    return vargo


# ============================================================================
# TEXT REPRESENTATION
# ============================================================================

def build_theme_text(row, vargo_map: dict) -> str:
    """
    Build rich text representation for GDELT theme.
    Format: "THEME_CODE - issue_category - description"
    """
    theme = row['theme_code']
    parts = [theme]
    
    # Add Vargo issue if available
    if theme in vargo_map:
        info = vargo_map[theme]
        if pd.notna(info.get('issue_category')):
            parts.append(str(info['issue_category']))
        if pd.notna(info.get('description')):
            parts.append(str(info['description']))
    
    return " - ".join(parts)


def prepare_theme_texts(themes: pd.DataFrame, vargo: pd.DataFrame) -> pd.DataFrame:
    """Prepare text representations for all themes."""
    
    # Build Vargo lookup
    vargo_map = {}
    if not vargo.empty:
        for _, row in vargo.iterrows():
            vargo_map[row['theme_code']] = {
                'issue_category': row.get('issue_category'),
                'description': row.get('description')
            }
    
    # Build text for each theme
    themes['text_repr'] = themes.apply(lambda r: build_theme_text(r, vargo_map), axis=1)
    themes['issue_category'] = themes['theme_code'].map(
        lambda t: vargo_map.get(t, {}).get('issue_category')
    )
    
    print(f"[*] Built text representations for {len(themes)} themes")
    print(f"    Sample: {themes['text_repr'].iloc[0][:80]}...")
    
    return themes


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embeddings(texts: list, model=None) -> np.ndarray:
    """
    Generate sentence embeddings using all-MiniLM-L6-v2.
    Falls back to random embeddings if model not available.
    """
    if HAS_SENTENCE_TRANSFORMERS and model is not None:
        print(f"[*] Encoding {len(texts)} texts with {MODEL_NAME}...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    else:
        print(f"[!] Using random embeddings (sentence-transformers not installed)")
        np.random.seed(42)
        return np.random.randn(len(texts), EMBEDDING_DIM).astype(np.float32)


def load_model():
    """Load sentence-transformer model."""
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading model on {device}...")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model


# ============================================================================
# IPTC MAPPING (SUPERVISED APPROACH)
# ============================================================================

def map_themes_to_iptc(
    theme_embeddings: np.ndarray,
    iptc_embeddings: np.ndarray,
    themes: pd.DataFrame,
    iptc: pd.DataFrame,
    top_level_only: bool = True
) -> pd.DataFrame:
    """
    Map each GDELT theme to nearest IPTC topic using cosine similarity.
    
    Args:
        theme_embeddings: GDELT theme embeddings (N x D)
        iptc_embeddings: IPTC topic embeddings (M x D)
        themes: GDELT themes DataFrame
        iptc: IPTC topics DataFrame
        top_level_only: If True, map only to top-level (17) IPTC categories
    
    Returns:
        themes DataFrame with IPTC mapping columns added
    """
    if not HAS_SKLEARN:
        print("[!] scikit-learn not available for similarity computation")
        themes['iptc_id'] = 'unknown'
        themes['iptc_label'] = 'Unknown'
        themes['similarity'] = 0.0
        return themes
    
    # Filter to top-level IPTC topics if requested
    if top_level_only:
        iptc_subset = iptc[iptc['level'] == 1].reset_index(drop=True)
        iptc_emb_subset = iptc_embeddings[iptc['level'] == 1]
    else:
        iptc_subset = iptc.reset_index(drop=True)
        iptc_emb_subset = iptc_embeddings
    
    print(f"[*] Mapping {len(themes)} themes to {len(iptc_subset)} IPTC topics...")
    
    # Compute cosine similarity matrix: themes × IPTC
    sim_matrix = cosine_similarity(theme_embeddings, iptc_emb_subset)
    
    # Find nearest IPTC topic for each theme
    closest_idx = np.argmax(sim_matrix, axis=1)
    max_sim = np.max(sim_matrix, axis=1)
    
    # Add mapping to themes DataFrame
    themes['iptc_id'] = iptc_subset.iloc[closest_idx]['medtop_id'].values
    themes['iptc_label'] = iptc_subset.iloc[closest_idx]['label'].values
    themes['iptc_definition'] = iptc_subset.iloc[closest_idx]['definition'].values
    themes['similarity'] = max_sim
    
    # Also get second-best match for comparison
    sim_matrix_copy = sim_matrix.copy()
    for i, idx in enumerate(closest_idx):
        sim_matrix_copy[i, idx] = -1  # mask best match
    second_idx = np.argmax(sim_matrix_copy, axis=1)
    second_sim = np.max(sim_matrix_copy, axis=1)
    
    themes['iptc_label_2nd'] = iptc_subset.iloc[second_idx]['label'].values
    themes['similarity_2nd'] = second_sim
    
    # Confidence: difference between best and second-best
    themes['confidence'] = themes['similarity'] - themes['similarity_2nd']
    
    return themes


# ============================================================================
# CLUSTERING (UNSUPERVISED APPROACH - OPTIONAL)
# ============================================================================

def cluster_themes(
    theme_embeddings: np.ndarray,
    themes: pd.DataFrame,
    n_clusters: int = 8
) -> pd.DataFrame:
    """
    Optional: Cluster themes using agglomerative clustering.
    This provides an alternative grouping independent of IPTC.
    """
    if not HAS_SKLEARN:
        themes['cluster_id'] = 0
        return themes
    
    print(f"[*] Clustering themes into {n_clusters} groups...")
    
    # PCA for dimensionality reduction
    n_components = min(20, len(themes) - 1, theme_embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(theme_embeddings)
    
    # Agglomerative clustering
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(X_reduced)
    
    themes['cluster_id'] = cluster_labels
    
    # Print cluster distribution
    for i in range(n_clusters):
        count = (cluster_labels == i).sum()
        print(f"   Cluster {i}: {count} themes")
    
    return themes


def compute_2d_projection(
    theme_embeddings: np.ndarray,
    themes: pd.DataFrame,
    method: str = 'tsne'
) -> pd.DataFrame:
    """
    Compute 2D projection of theme embeddings for visualization.
    
    Args:
        theme_embeddings: High-dimensional embeddings (N x D)
        themes: DataFrame with theme information
        method: 'tsne' or 'pca'
    
    Returns:
        themes DataFrame with tsne_x, tsne_y columns added
    """
    if not HAS_SKLEARN:
        themes['tsne_x'] = np.random.randn(len(themes)) * 10 + 50
        themes['tsne_y'] = np.random.randn(len(themes)) * 10 + 50
        return themes
    
    n_samples = len(themes)
    print(f"[*] Computing 2D {method.upper()} projection for {n_samples} themes...")
    
    if n_samples < 5:
        themes['tsne_x'] = np.random.randn(n_samples) * 10 + 50
        themes['tsne_y'] = np.random.randn(n_samples) * 10 + 50
        return themes
    
    if method == 'tsne':
        # t-SNE with optimized parameters
        perplexity = min(30, n_samples - 1)
        
        # PCA pre-reduction for faster t-SNE
        if theme_embeddings.shape[1] > 50:
            n_pca = min(50, n_samples - 1)
            pca = PCA(n_components=n_pca, random_state=42)
            embeddings_reduced = pca.fit_transform(theme_embeddings)
            print(f"   PCA pre-reduction: {theme_embeddings.shape[1]} -> {n_pca} dims")
        else:
            embeddings_reduced = theme_embeddings
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca' if n_samples > 50 else 'random',
            random_state=42,
            max_iter=1000
        )
        coords_2d = tsne.fit_transform(embeddings_reduced)
    else:
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(theme_embeddings)
    
    # Normalize to 0-100 range
    coords_min = coords_2d.min(axis=0)
    coords_max = coords_2d.max(axis=0)
    coords_range = coords_max - coords_min
    coords_range[coords_range == 0] = 1
    
    coords_normalized = 5 + 90 * (coords_2d - coords_min) / coords_range
    
    themes['tsne_x'] = coords_normalized[:, 0]
    themes['tsne_y'] = coords_normalized[:, 1]
    
    print(f"   2D projection complete. Range: x=[{coords_normalized[:,0].min():.1f}, {coords_normalized[:,0].max():.1f}], y=[{coords_normalized[:,1].min():.1f}, {coords_normalized[:,1].max():.1f}]")
    
    return themes


# ============================================================================
# ANALYSIS & EXPORT
# ============================================================================

def analyze_mapping(themes: pd.DataFrame, iptc: pd.DataFrame) -> dict:
    """Analyze IPTC mapping results."""
    
    # Count themes per IPTC category
    iptc_counts = themes['iptc_label'].value_counts().to_dict()
    
    # Average similarity per IPTC category
    iptc_sim = themes.groupby('iptc_label')['similarity'].mean().to_dict()
    
    # Low confidence mappings (ambiguous)
    low_conf = themes[themes['confidence'] < 0.1]
    
    # Build summary
    summary = {
        'total_themes': len(themes),
        'iptc_categories_used': len(iptc_counts),
        'avg_similarity': float(themes['similarity'].mean()),
        'avg_confidence': float(themes['confidence'].mean()),
        'low_confidence_count': len(low_conf),
        'themes_per_iptc': iptc_counts,
        'avg_similarity_per_iptc': iptc_sim
    }
    
    return summary


def export_results(themes: pd.DataFrame, iptc: pd.DataFrame, output_dir: Path):
    """Export mapping results to JSON and CSV."""
    
    # Analyze mapping
    summary = analyze_mapping(themes, iptc)
    
    # Prepare IPTC category details
    iptc_top = iptc[iptc['level'] == 1].copy()
    iptc_details = {}
    
    for _, row in iptc_top.iterrows():
        label = row['label']
        matched = themes[themes['iptc_label'] == label]
        
        iptc_details[label] = {
            'medtop_id': row['medtop_id'],
            'definition': row['definition'],
            'theme_count': len(matched),
            'themes': matched['theme_code'].tolist(),
            'avg_similarity': float(matched['similarity'].mean()) if len(matched) > 0 else 0
        }
    
    # Convert NaN values to None for JSON serialization
    def clean_value(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.floating, float)):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v
    
    # Prepare themes list
    themes_list = []
    for _, row in themes.iterrows():
        themes_list.append({
            'theme_code': row['theme_code'],
            'issue_category': clean_value(row.get('issue_category')),
            'iptc_id': row['iptc_id'],
            'iptc_label': row['iptc_label'],
            'iptc_definition': row['iptc_definition'],
            'similarity': clean_value(row['similarity']),
            'iptc_label_2nd': row['iptc_label_2nd'],
            'similarity_2nd': clean_value(row['similarity_2nd']),
            'confidence': clean_value(row['confidence']),
            'cluster_id': clean_value(row.get('cluster_id', 0)),
            # 2D coordinates for visualization
            'tsne_x': clean_value(row.get('tsne_x', 50)),
            'tsne_y': clean_value(row.get('tsne_y', 50))
        })
    
    # Build final JSON
    result = {
        'metadata': {
            'model': MODEL_NAME,
            'embedding_dim': EMBEDDING_DIM,
            'method': 'IPTC Media Topics cosine similarity mapping',
            'reference': 'IPTC NewsCodes 2025-10-10',
            'total_themes': summary['total_themes'],
            'iptc_categories': summary['iptc_categories_used']
        },
        'summary': summary,
        'iptc_categories': iptc_details,
        'themes': themes_list
    }
    
    # Export JSON
    json_file = output_dir / OUTPUT_JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Exported: {json_file}")
    
    # Export CSV
    csv_file = output_dir / OUTPUT_CSV
    themes.to_csv(csv_file, index=False)
    print(f"[OK] Exported: {csv_file}")
    
    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_iptc_mapping_pipeline(data_dir: Path = None) -> dict:
    """Run the complete IPTC mapping pipeline."""
    
    if data_dir is None:
        data_dir = Path(__file__).parent
    
    print("\n" + "=" * 70)
    print("GDELT Theme -> IPTC Media Topics Mapping Pipeline (V1)")
    print("=" * 70 + "\n")
    
    # Step 1: Load IPTC topics
    print("[1/8] Loading IPTC Media Topics...")
    iptc = load_iptc_topics(data_dir)
    
    # Step 2: Load GDELT themes
    print("\n[2/8] Loading GDELT themes...")
    themes = load_gdelt_themes(data_dir)
    
    # Step 3: Load Vargo mapping for additional context
    print("\n[3/8] Loading Vargo issue mapping...")
    vargo = load_vargo_mapping(data_dir)
    
    # Step 4: Prepare text representations
    print("\n[4/8] Building text representations...")
    themes = prepare_theme_texts(themes, vargo)
    
    # Step 5: Load model and generate embeddings
    print("\n[5/8] Generating embeddings...")
    model = load_model()
    
    # IPTC embeddings
    iptc_texts = iptc['text_repr'].tolist()
    iptc_embeddings = generate_embeddings(iptc_texts, model)
    
    # Theme embeddings
    theme_texts = themes['text_repr'].tolist()
    theme_embeddings = generate_embeddings(theme_texts, model)
    
    # Step 6: Map themes to IPTC
    print("\n[6/8] Mapping themes to IPTC categories...")
    themes = map_themes_to_iptc(
        theme_embeddings, iptc_embeddings, themes, iptc, top_level_only=True
    )
    
    # Step 7: Optional clustering
    print("\n[7/8] Additional clustering...")
    n_clusters = min(8, len(themes))
    themes = cluster_themes(theme_embeddings, themes, n_clusters=n_clusters)
    
    # Step 7b: Compute 2D projection for visualization
    print("\n[7b/8] Computing 2D projection for visualization...")
    themes = compute_2d_projection(theme_embeddings, themes, method='tsne')
    
    # Step 8: Export results
    print("\n[8/8] Exporting results...")
    result = export_results(themes, iptc, data_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("MAPPING RESULTS SUMMARY")
    print("=" * 70)
    
    summary = result['summary']
    print(f"\n  Total themes mapped: {summary['total_themes']}")
    print(f"  IPTC categories used: {summary['iptc_categories_used']}")
    print(f"  Average similarity: {summary['avg_similarity']:.3f}")
    print(f"  Average confidence: {summary['avg_confidence']:.3f}")
    print(f"  Low confidence mappings: {summary['low_confidence_count']}")
    
    print("\nThemes per IPTC Category:")
    for iptc_label, count in sorted(summary['themes_per_iptc'].items(), key=lambda x: -x[1]):
        avg_sim = summary['avg_similarity_per_iptc'].get(iptc_label, 0)
        print(f"  - {iptc_label}: {count} themes (avg sim: {avg_sim:.3f})")
    
    print("\n[OK] Pipeline complete!")
    
    return result


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map GDELT themes to IPTC Media Topics"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing data files"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent
    
    results = run_iptc_mapping_pipeline(data_dir)
