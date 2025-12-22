#!/usr/bin/env python3
"""
GDELT Theme -> IPTC Media Topics Mapper (v2)
Two-Layer Approach: Rule-based + Embedding-based Fusion

Layer 1: Rule-based mapping using theme prefixes + Vargo issue taxonomy
Layer 2: Semantic embedding with Sentence-BERT + cosine similarity
Fusion: Combines both layers with confidence thresholds

Author: Omer Alper Guzel
Date: December 2024
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime

# Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[!] sentence-transformers not installed. Run: pip install sentence-transformers")

# ============================================================================
# IPTC MEDIA TOPICS - 17 TOP-LEVEL CATEGORIES
# ============================================================================

IPTC_CATEGORIES = [
    {"id": "01000000", "label": "arts, culture, entertainment and media", 
     "definition": "All forms of arts, entertainment, cultural heritage and media"},
    {"id": "02000000", "label": "crime, law and justice",
     "definition": "Crime, legal proceedings, police, courts, punishment"},
    {"id": "03000000", "label": "disaster, accident and emergency incident",
     "definition": "Natural and man-made disasters, accidents, emergency response"},
    {"id": "04000000", "label": "economy, business and finance",
     "definition": "Economic activity, business, markets, trade, financial matters"},
    {"id": "05000000", "label": "education",
     "definition": "Education systems, schools, universities, learning"},
    {"id": "06000000", "label": "environment",
     "definition": "Environmental issues, climate, conservation, pollution"},
    {"id": "07000000", "label": "health",
     "definition": "Physical and mental health, diseases, healthcare systems"},
    {"id": "08000000", "label": "human interest",
     "definition": "Human interest stories, curiosities, personal stories"},
    {"id": "09000000", "label": "labour",
     "definition": "Employment, working conditions, labour relations, unions"},
    {"id": "10000000", "label": "lifestyle and leisure",
     "definition": "Lifestyle, leisure activities, travel, fashion, food"},
    {"id": "11000000", "label": "politics and government",
     "definition": "Politics, government, elections, public policy"},
    {"id": "12000000", "label": "religion",
     "definition": "Religious beliefs, practices, institutions, faith"},
    {"id": "13000000", "label": "science and technology",
     "definition": "Scientific research, technological developments, innovation"},
    {"id": "14000000", "label": "society",
     "definition": "Social issues, demographics, family, social welfare"},
    {"id": "15000000", "label": "sport",
     "definition": "Sports events, competitions, athletes, sports organizations"},
    {"id": "16000000", "label": "conflict, war and peace",
     "definition": "Armed conflicts, wars, peace processes, military operations"},
    {"id": "17000000", "label": "weather",
     "definition": "Weather conditions, forecasts, climate patterns"}
]

# ============================================================================
# LAYER 1: RULE-BASED MAPPING (PREFIX + PATTERN)
# ============================================================================

# Prefix patterns -> IPTC ID mapping (EXTENDED)
RULE_PATTERNS = {
    # Economy, Business and Finance (04)
    r'^(ECON|TAX|WB|EPU|FISCAL|DEBT|INFLATION|TRADE|MARKET|BANK|CURRENCY|STOCK|GDP|INVEST|BUDGET|SUBSID|AUSTERITY|FUELPRICES|INCOME|PRIVATIZATION)': '04000000',
    
    # Politics and Government (11)
    r'^(POLITICAL|GOV|POL|DEMOCRACY|ELECTION|VOTE|PARLIAMENT|LEGISLATION|POLICY|DIPLOMACY|TREATY|SANCTION|LEADER|REFERENDUM|UNGP|VETO|SOVEREIGNTY|STATE|PUBLIC|FREESPEECH|TRANSPARENCY|UNGOVERNED|PROTEST|CONSTITUTIONAL)': '11000000',
    
    # Conflict, War and Peace (16)
    r'^(ARMEDCONFLICT|TERROR|MIL|MILITARY|WMD|WAR|PEACE|REBEL|REBELLION|REBELS|INSURGENT|INSURGENCY|GENOCIDE|ETHNIC_VIOLENCE|CEASEFIRE|KILL|CRISISLEX|SECURITY|ALLIANCE|BLOCKADE|CHECKPOINT|JIHAD|SEPARATIST|EXTREMISM|UNREST|MARITIME|DRONES|NEGOTIATIONS|PEACEKEEPING)': '16000000',
    
    # Crime, Law and Justice (02)
    r'^(CRIME|KIDNAP|RAPE|DRUG|ARREST|PRISON|COURT|TRIAL|VERDICT|CORRUPTION|FRAUD|THEFT|MURDER|ASSAULT|SMUGGL|SMUGGLING|ASSASSINATION|BULLYING|IMPEACHMENT|JUSTICE|LEGALIZE|PIRACY|SEIZE|TORTURE|TREASON|TRAFFICKING|VIOLENT|HARASSMENT|RETALIATE|SURVEILLANCE|VANDALIZE|WHISTLEBLOWER)': '02000000',
    
    # Disaster, Accident and Emergency (03)
    r'^(DISASTER|NATURAL|FLOOD|EARTHQUAKE|TSUNAMI|HURRICANE|TORNADO|FIRE|FIREARM|ACCIDENT|EMERGENCY|MANMADE|FAMINE|DROUGHT|EVACUATION|LANDMINE|LOCUSTS|UNSAFE|WOUND)': '03000000',
    
    # Health (07)
    r'^(HEALTH|MEDICAL|MED|DISEASE|PANDEMIC|EPIDEMIC|VACCINE|HOSPITAL|SANITATION|MENTAL_HEALTH|DRUG_USE|HIV|COVID|EBOLA|DISABILITY|SICKENED|SUICIDE)': '07000000',
    
    # Education (05)
    r'^(EDUCATION|SCHOOL|UNIVERSITY|STUDENT|TEACHER|LITERACY|CURRICULUM|INFRASTRUCTURE)': '05000000',
    
    # Environment (06)
    r'^(ENV|CLIMATE|POLLUTION|CONSERVATION|WILDLIFE|DEFOREST|CARBON|EMISSION|BIODIVERSITY|WATER|EXHUMATION)': '06000000',
    
    # Society (14)
    r'^(SOC|POVERTY|HUMAN_RIGHTS|LGBT|GENDER|DISCRIMINATION|REFUGEE|REFUGEES|MIGRANT|HOMELESS|CHILD_|WOMEN_|MINORITY|INEQUALITY|POPULATION|RURAL|URBAN|GENTRIFICATION|IMMIGRATION|SLUMS|CURFEW|DISPLACED)': '14000000',
    
    # Religion (12)
    r'^(RELIGION|CHURCH|MOSQUE|TEMPLE|FAITH|CHRISTIAN|MUSLIM|HINDU|BUDDHIST|JEWISH|POPE|CLERGY|IDEOLOGY|PERSECUTION)': '12000000',
    
    # Sport (15)
    r'^(SPORT|OLYMPIC|FOOTBALL|SOCCER|BASKETBALL|TENNIS|CRICKET|RUGBY|ATHLETICS|FIFA|UEFA)': '15000000',
    
    # Arts, Culture, Entertainment and Media (01)
    r'^(MEDIA|INFO|PROPAGANDA|CENSOR|JOURNAL|NEWS|BROADCAST|CULTURE|ART|MUSIC|FILM|THEATER|MUSEUM|HERITAGE)': '01000000',
    
    # Lifestyle and Leisure (10)
    r'^(TOURISM|TRAVEL|LEISURE|FASHION|FOOD|RESTAURANT|HOTEL|VACATION|ENTERTAINMENT|CELEBRITY|MOVEMENT|RETIREMENT|RETIREMENTS|EXILE)': '10000000',
    
    # Labour (09)
    r'^(LABOR|LABOUR|EMPLOYMENT|UNEMPLOY|UNEMPLOYMENT|STRIKE|UNION|WAGE|WORKER|JOB_|RECRUITMENT|GRIEVANCES|SHORTAGE|RELATIONS|RAIL|BORDER|CLAIM|RESIGNATION|APPOINTMENT)': '09000000',
    
    # Science and Technology (13)
    r'^(SCIENCE|TECH|RESEARCH|INNOVATION|SPACE|CYBER|INTERNET|AI|ROBOT|BIOTECH|NANOTECH|AVIATION|PHONE|PIPELINE|NEW)': '13000000',
    
    # Weather (17)
    r'^(WEATHER|FORECAST|STORM|RAIN|SNOW|TEMPERATURE|WIND|CLOUD)': '17000000',
    
    # Human Interest (08)
    r'^(HUMAN_INTEREST|CURIOSITY|PERSONAL|LIFESTYLE|AFFECT|EMOTION|HUMAN|SELF|CHARASMATIC)': '08000000',
    
    # Additional specific mappings
    r'^(DEATH|HATE|SCANDAL|BAN|SEIGE|CONFISCATION|PROPERTY)': '02000000',
    r'^(ORGANIZED|TAKE|LEG)': '15000000',
    r'^(ROAD|TRAFFIC|POWER)': '11000000',
    r'^(ETH|GEN|REL|CRM|SLFID|USPEC)': '14000000',
    r'^(EMERG|DELAY|DEFECTION|CLOSURE)': '03000000',
    r'^(RELEASE|RATIFY)': '11000000',
    r'^(BLACK)': '14000000',
}

def apply_rule_based_mapping(theme_code: str) -> tuple:
    """
    Apply rule-based mapping to a theme code.
    Returns (iptc_id, confidence) or (None, 0)
    """
    if not theme_code:
        return None, 0
    
    theme_upper = theme_code.upper()
    
    for pattern, iptc_id in RULE_PATTERNS.items():
        if re.match(pattern, theme_upper):
            confidence = 0.9 if len(pattern) > 20 else 0.8
            return iptc_id, confidence
    
    return None, 0

# ============================================================================
# LAYER 2: EMBEDDING-BASED MAPPING
# ============================================================================

def load_vargo_mapping(filepath: str = "vargo_gdelt_themes_issues.csv") -> dict:
    """Load Vargo's GDELT theme to issue mapping"""
    vargo_map = {}
    try:
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            vargo_map[row['theme_code']] = {
                'issue': row['issue_category'],
                'description': row.get('description', '')
            }
    except Exception as e:
        print(f"[!] Could not load Vargo mapping: {e}")
    return vargo_map

def build_theme_text(theme_code: str, vargo_map: dict) -> str:
    """Build rich text representation for a theme"""
    parts = [theme_code]
    
    if theme_code in vargo_map:
        info = vargo_map[theme_code]
        if info.get('issue'):
            parts.append(info['issue'])
        if info.get('description'):
            parts.append(info['description'])
    
    readable = theme_code.replace('_', ' ').replace('-', ' ').title()
    if readable != theme_code:
        parts.append(readable)
    
    return ' - '.join(parts)

def build_iptc_texts(categories: list) -> list:
    """Build text representations for IPTC categories"""
    return [f"{cat['label']} - {cat['definition']}" for cat in categories]

def compute_embeddings(texts: list, model, batch_size: int = 64) -> np.ndarray:
    """Compute sentence embeddings"""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

def find_nearest_iptc(theme_emb: np.ndarray, iptc_emb: np.ndarray, categories: list) -> list:
    """Find nearest IPTC category for each theme embedding"""
    sim = cosine_similarity(theme_emb, iptc_emb)
    
    results = []
    for i in range(len(sim)):
        scores = sim[i]
        best_idx = np.argmax(scores)
        second_best_idx = np.argsort(scores)[-2]
        
        results.append({
            'nn_id': categories[best_idx]['id'],
            'nn_label': categories[best_idx]['label'],
            'nn_score': float(scores[best_idx]),
            'second_id': categories[second_best_idx]['id'],
            'second_label': categories[second_best_idx]['label'],
            'second_score': float(scores[second_best_idx])
        })
    
    return results


def compute_2d_projection(embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:
    """
    Compute 2D projection of embeddings using t-SNE or PCA.
    
    Args:
        embeddings: High-dimensional embeddings (N x D)
        method: 'tsne' or 'pca'
    
    Returns:
        2D coordinates (N x 2)
    """
    n_samples = len(embeddings)
    
    if n_samples < 5:
        print(f"  [!] Too few samples ({n_samples}) for dimensionality reduction")
        return np.random.randn(n_samples, 2) * 10
    
    if method == 'tsne':
        # t-SNE parameters optimized for theme clustering visualization
        perplexity = min(30, n_samples - 1)  # perplexity must be < n_samples
        print(f"  Running t-SNE (perplexity={perplexity})...")
        
        # First reduce with PCA if high dimensional (faster t-SNE)
        if embeddings.shape[1] > 50:
            n_pca = min(50, n_samples - 1)
            pca = PCA(n_components=n_pca, random_state=42)
            embeddings_reduced = pca.fit_transform(embeddings)
            print(f"  PCA pre-reduction: {embeddings.shape[1]} -> {n_pca} dims")
        else:
            embeddings_reduced = embeddings
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca' if n_samples > 50 else 'random',
            random_state=42,
            max_iter=1000
        )
        coords_2d = tsne.fit_transform(embeddings_reduced)
        
    else:  # PCA
        print("  Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embeddings)
    
    # Normalize to 0-100 range for visualization
    coords_min = coords_2d.min(axis=0)
    coords_max = coords_2d.max(axis=0)
    coords_range = coords_max - coords_min
    coords_range[coords_range == 0] = 1  # Avoid division by zero
    
    coords_normalized = 5 + 90 * (coords_2d - coords_min) / coords_range  # 5-95 range
    
    return coords_normalized

# ============================================================================
# FUSION: COMBINE RULE-BASED AND EMBEDDING-BASED
# ============================================================================

THR_STRONG = 0.45
THR_WEAK = 0.30

def fusion_decision(rule_id: str, rule_conf: float, nn_id: str, nn_score: float) -> tuple:
    """
    Fusion logic to combine rule-based and embedding-based predictions.
    Returns (final_iptc_id, decision_source, confidence)
    """
    if rule_id and rule_id == nn_id:
        return rule_id, 'agreement', max(rule_conf, nn_score)
    
    if not rule_id:
        if nn_score >= THR_STRONG:
            return nn_id, 'nn_confident', nn_score
        elif nn_score >= THR_WEAK:
            return nn_id, 'nn_weak', nn_score
        else:
            return None, 'unclassified', nn_score
    
    if nn_score >= THR_STRONG:
        return nn_id, 'nn_override', nn_score
    elif nn_score <= THR_WEAK:
        return rule_id, 'rule_preferred', rule_conf
    else:
        return rule_id, 'ambiguous', (rule_conf + nn_score) / 2

def get_iptc_label(iptc_id: str) -> str:
    """Get IPTC category label from ID"""
    for cat in IPTC_CATEGORIES:
        if cat['id'] == iptc_id:
            return cat['label']
    return ''

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_mapping(output_dir: str = ".") -> dict:
    """
    Run the complete two-layer GDELT -> IPTC mapping pipeline.
    
    Returns:
        Dictionary with mapping results and statistics
    """
    print("=" * 60)
    print("GDELT Theme -> IPTC Media Topics Mapper (v2)")
    print("Two-Layer Approach: Rule-based + Embedding Fusion")
    print("=" * 60)
    
    output_path = Path(output_dir)
    
    # Load theme codes from all CSV files
    print("\n[*] Loading theme codes from CSV files...")
    
    theme_codes = set()
    csv_files = [
        'gdelt_top15_themes_by_country_2022_2024.csv',
        'gdelt_monthly_quality_metrics.csv',
        'gdelt_monthly_docs_per_theme_country_2022_2024.csv'
    ]
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(output_path / csv_file)
            if 'theme_code' in df.columns:
                theme_codes.update(df['theme_code'].dropna().unique())
        except Exception as e:
            print(f"  [!] Could not read {csv_file}: {e}")
    
    if not theme_codes:
        print("[X] No theme codes found in CSV files!")
        return None
    
    print(f"  [OK] Found {len(theme_codes)} unique theme codes")
    
    # Load Vargo mapping
    print("\n[*] Loading Vargo issue mapping...")
    vargo_map = load_vargo_mapping(output_path / "vargo_gdelt_themes_issues.csv")
    print(f"  [OK] Loaded {len(vargo_map)} Vargo mappings")
    
    # LAYER 1: Rule-based mapping
    print("\n[*] Layer 1: Applying rule-based mapping...")
    rule_results = {}
    rule_matched = 0
    
    for theme in theme_codes:
        iptc_id, confidence = apply_rule_based_mapping(theme)
        rule_results[theme] = {'rule_id': iptc_id, 'rule_conf': confidence}
        if iptc_id:
            rule_matched += 1
    
    print(f"  [OK] Rule-based matched: {rule_matched}/{len(theme_codes)} ({100*rule_matched/len(theme_codes):.1f}%)")
    
    # LAYER 2: Embedding-based mapping
    if not TRANSFORMERS_AVAILABLE:
        print("\n[!] Sentence-transformers not available. Using rule-based only.")
        nn_results = {theme: {'nn_id': None, 'nn_label': '', 'nn_score': 0} for theme in theme_codes}
    else:
        print("\n[*] Layer 2: Computing semantic embeddings...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")
        
        print("  Loading sentence-transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Build text representations
        theme_list = sorted(theme_codes)
        theme_texts = [build_theme_text(t, vargo_map) for t in theme_list]
        iptc_texts = build_iptc_texts(IPTC_CATEGORIES)
        
        print(f"  Encoding {len(theme_texts)} themes...")
        theme_embeddings = compute_embeddings(theme_texts, model)
        
        print(f"  Encoding {len(iptc_texts)} IPTC categories...")
        iptc_embeddings = compute_embeddings(iptc_texts, model)
        
        print("  Computing cosine similarities...")
        nn_list = find_nearest_iptc(theme_embeddings, iptc_embeddings, IPTC_CATEGORIES)
        
        nn_results = {theme: nn_list[i] for i, theme in enumerate(theme_list)}
        
        # Compute 2D projections for visualization
        print("\n[*] Computing 2D projections for visualization...")
        coords_2d = compute_2d_projection(theme_embeddings, method='tsne')
        
        # Store coordinates in nn_results
        for i, theme in enumerate(theme_list):
            nn_results[theme]['tsne_x'] = float(coords_2d[i, 0])
            nn_results[theme]['tsne_y'] = float(coords_2d[i, 1])
    
    # FUSION
    print("\n[*] Fusion: Combining rule-based and embedding results...")
    
    final_results = []
    decision_counts = {}
    
    for theme in sorted(theme_codes):
        rule_id = rule_results[theme]['rule_id']
        rule_conf = rule_results[theme]['rule_conf']
        
        nn_data = nn_results.get(theme, {'nn_id': None, 'nn_label': '', 'nn_score': 0})
        nn_id = nn_data.get('nn_id')
        nn_score = nn_data.get('nn_score', 0)
        
        final_id, decision, confidence = fusion_decision(rule_id, rule_conf, nn_id, nn_score)
        
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        result = {
            'theme_code': theme,
            'text_repr': build_theme_text(theme, vargo_map),
            'iptc_rule_id': rule_id,
            'iptc_rule_label': get_iptc_label(rule_id) if rule_id else '',
            'rule_confidence': rule_conf,
            'iptc_nn_id': nn_id,
            'iptc_nn_label': nn_data.get('nn_label', ''),
            'nn_score': nn_score,
            'iptc_second_id': nn_data.get('second_id', ''),
            'iptc_second_label': nn_data.get('second_label', ''),
            'second_score': nn_data.get('second_score', 0),
            'iptc_final_id': final_id,
            'iptc_final_label': get_iptc_label(final_id) if final_id else '',
            'decision_source': decision,
            'final_confidence': confidence,
            # 2D coordinates for visualization
            'tsne_x': nn_data.get('tsne_x', 50 + np.random.randn() * 10),
            'tsne_y': nn_data.get('tsne_y', 50 + np.random.randn() * 10)
        }
        final_results.append(result)
    
    # Statistics
    print("\n[*] Fusion Statistics:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(theme_codes)
        print(f"  {decision}: {count} ({pct:.1f}%)")
    
    # Category distribution
    category_dist = {}
    for r in final_results:
        cat = r['iptc_final_label'] or 'unclassified'
        category_dist[cat] = category_dist.get(cat, 0) + 1
    
    print("\n[*] IPTC Category Distribution:")
    for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # Save results
    print("\n[*] Saving results...")
    
    # Build output structure
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_themes": len(theme_codes),
            "rule_matched": rule_matched,
            "iptc_categories": len(set(r['iptc_final_label'] for r in final_results if r['iptc_final_label'])),
            "decision_stats": decision_counts
        },
        "themes": final_results,
        "iptc_categories": {}
    }
    
    # Group by IPTC category
    for r in final_results:
        cat_label = r['iptc_final_label'] or 'unclassified'
        if cat_label not in output_data["iptc_categories"]:
            output_data["iptc_categories"][cat_label] = {
                "iptc_id": r['iptc_final_id'] or '',
                "theme_count": 0,
                "themes": []
            }
        output_data["iptc_categories"][cat_label]["theme_count"] += 1
        output_data["iptc_categories"][cat_label]["themes"].append(r['theme_code'])
    
    # Save JSON
    json_path = output_path / "gdelt_iptc_mapping_v2.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  [OK] JSON: {json_path.name}")
    
    # Save CSV
    csv_path = output_path / "gdelt_themes_iptc_v2.csv"
    pd.DataFrame(final_results).to_csv(csv_path, index=False)
    print(f"  [OK] CSV: {csv_path.name}")
    
    print("\n[OK] Mapping complete!")
    print(f"\n[*] Summary:")
    print(f"   Total themes: {len(theme_codes)}")
    print(f"   Agreement rate: {decision_counts.get('agreement', 0)}")
    print(f"   Unclassified: {decision_counts.get('unclassified', 0)}")
    
    return output_data

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    results = run_full_mapping(output_dir=output_dir)
