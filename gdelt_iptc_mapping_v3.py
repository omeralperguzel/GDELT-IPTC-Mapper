#!/usr/bin/env python3
"""
GDELT Theme -> IPTC Media Topics Mapper (v3)
Hierarchical Approach: Top-level + Subtopic Definitions

Layer 1: Rule-based mapping using theme prefixes + Vargo issue taxonomy
Layer 2: Semantic embedding with Sentence-BERT + cosine similarity
Layer 3: Subtopic refinement with enhanced similarity scoring
Fusion: Hierarchical decision tree (top-level → subtopic)

Author: Omer Alper Guzel
Date: January 2026
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
# IPTC MEDIA TOPICS - 17 TOP-LEVEL CATEGORIES + SUBTOPICS (2025-10-10 Standard)
# ============================================================================

IPTC_CATEGORIES_V3 = [
    {
        "id": "01000000",
        "label": "arts, culture, entertainment and media",
        "definition": "All forms of arts, entertainment, cultural heritage and media",
        "subtopics": [
            {"id": "01010000", "label": "arts", "definition": "Visual arts, performing arts, literature"},
            {"id": "01020000", "label": "culture", "definition": "Cultural heritage, traditions, cultural events"},
            {"id": "01030000", "label": "entertainment", "definition": "Entertainment industry, leisure activities"},
            {"id": "01040000", "label": "media", "definition": "Journalism, broadcasting, digital media"}
        ]
    },
    {
        "id": "02000000",
        "label": "crime, law and justice",
        "definition": "Crime, legal proceedings, police, courts, punishment",
        "subtopics": [
            {"id": "02010000", "label": "crime", "definition": "Criminal activities, offenses"},
            {"id": "02020000", "label": "law enforcement", "definition": "Police, investigations, security"},
            {"id": "02030000", "label": "justice system", "definition": "Courts, trials, legal processes"}
        ]
    },
    {
        "id": "03000000",
        "label": "disaster, accident and emergency incident",
        "definition": "Natural and man-made disasters, accidents, emergency response",
        "subtopics": [
            {"id": "03010000", "label": "natural disaster", "definition": "Earthquakes, floods, hurricanes, wildfires"},
            {"id": "03020000", "label": "man-made disaster", "definition": "Industrial accidents, explosions, structural failures"},
            {"id": "03030000", "label": "emergency response", "definition": "Rescue operations, evacuation, humanitarian aid"}
        ]
    },
    {
        "id": "04000000",
        "label": "economy, business and finance",
        "definition": "Economic activity, business, markets, trade, financial matters",
        "subtopics": [
            {"id": "04010000", "label": "business", "definition": "Corporate activities, entrepreneurship, commerce"},
            {"id": "04020000", "label": "finance", "definition": "Banking, investments, financial markets"},
            {"id": "04030000", "label": "economy", "definition": "Economic indicators, trade, employment"}
        ]
    },
    {
        "id": "05000000",
        "label": "education",
        "definition": "Education systems, schools, universities, learning",
        "subtopics": [
            {"id": "05010000", "label": "primary education", "definition": "Elementary and secondary schooling"},
            {"id": "05020000", "label": "higher education", "definition": "Universities, colleges, postgraduate studies"},
            {"id": "05030000", "label": "vocational training", "definition": "Technical education, apprenticeships"}
        ]
    },
    {
        "id": "06000000",
        "label": "environment",
        "definition": "Environmental issues, climate, conservation, pollution",
        "subtopics": [
            {"id": "06010000", "label": "climate change", "definition": "Global warming, emissions, climate policy"},
            {"id": "06020000", "label": "conservation", "definition": "Wildlife protection, biodiversity, habitats"},
            {"id": "06030000", "label": "pollution", "definition": "Air, water, soil contamination"}
        ]
    },
    {
        "id": "07000000",
        "label": "health",
        "definition": "Physical and mental health, diseases, healthcare systems",
        "subtopics": [
            {"id": "07010000", "label": "medical research", "definition": "Drug development, clinical trials"},
            {"id": "07020000", "label": "public health", "definition": "Disease prevention, health policy"},
            {"id": "07030000", "label": "healthcare", "definition": "Hospitals, medical services, insurance"}
        ]
    },
    {
        "id": "08000000",
        "label": "human interest",
        "definition": "Human interest stories, curiosities, personal stories",
        "subtopics": [
            {"id": "08010000", "label": "personal stories", "definition": "Individual experiences, biographies"},
            {"id": "08020000", "label": "curiosities", "definition": "Unusual events, remarkable achievements"},
            {"id": "08030000", "label": "human condition", "definition": "Human experiences, emotions, relationships"}
        ]
    },
    {
        "id": "09000000",
        "label": "labour",
        "definition": "Employment, working conditions, labour relations, unions",
        "subtopics": [
            {"id": "09010000", "label": "employment", "definition": "Job creation, unemployment, workforce"},
            {"id": "09020000", "label": "working conditions", "definition": "Wages, safety, workplace rights"},
            {"id": "09030000", "label": "labour relations", "definition": "Unions, collective bargaining, strikes"}
        ]
    },
    {
        "id": "10000000",
        "label": "lifestyle and leisure",
        "definition": "Lifestyle, leisure activities, travel, fashion, food",
        "subtopics": [
            {"id": "10010000", "label": "travel", "definition": "Tourism, vacations, transportation"},
            {"id": "10020000", "label": "fashion", "definition": "Clothing, style, designers"},
            {"id": "10030000", "label": "food and dining", "definition": "Cuisine, restaurants, culinary arts"}
        ]
    },
    {
        "id": "11000000",
        "label": "politics and government",
        "definition": "Politics, government, elections, public policy",
        "subtopics": [
            {"id": "11010000", "label": "elections", "definition": "Voting, campaigns, political parties"},
            {"id": "11020000", "label": "government", "definition": "Public administration, policy making"},
            {"id": "11030000", "label": "international relations", "definition": "Diplomacy, treaties, foreign policy"}
        ]
    },
    {
        "id": "12000000",
        "label": "religion",
        "definition": "Religious beliefs, practices, institutions, faith",
        "subtopics": [
            {"id": "12010000", "label": "religious institutions", "definition": "Churches, mosques, temples, clergy"},
            {"id": "12020000", "label": "religious practices", "definition": "Rituals, ceremonies, pilgrimages"},
            {"id": "12030000", "label": "religious beliefs", "definition": "Faith, theology, spiritual matters"}
        ]
    },
    {
        "id": "13000000",
        "label": "science and technology",
        "definition": "Scientific research, technological developments, innovation",
        "subtopics": [
            {"id": "13010000", "label": "scientific research", "definition": "Basic research, discoveries, academia"},
            {"id": "13020000", "label": "technology", "definition": "Digital technology, engineering, innovation"},
            {"id": "13030000", "label": "space exploration", "definition": "Astronomy, space missions, cosmology"}
        ]
    },
    {
        "id": "14000000",
        "label": "society",
        "definition": "Social issues, demographics, family, social welfare",
        "subtopics": [
            {"id": "14010000", "label": "social welfare", "definition": "Poverty, inequality, social services"},
            {"id": "14020000", "label": "demographics", "definition": "Population, migration, urbanization"},
            {"id": "14030000", "label": "family", "definition": "Marriage, children, family life"}
        ]
    },
    {
        "id": "15000000",
        "label": "sport",
        "definition": "Sports events, competitions, athletes, sports organizations",
        "subtopics": [
            {"id": "15010000", "label": "professional sport", "definition": "Professional leagues, athletes, competitions"},
            {"id": "15020000", "label": "amateur sport", "definition": "Recreational sports, Olympics, youth sports"},
            {"id": "15030000", "label": "sports organizations", "definition": "Federations, governing bodies, events"}
        ]
    },
    {
        "id": "16000000",
        "label": "conflict, war and peace",
        "definition": "Armed conflicts, wars, peace processes, military operations",
        "subtopics": [
            {"id": "16010000", "label": "armed conflict", "definition": "Battles, military operations, casualties"},
            {"id": "16020000", "label": "peace processes", "definition": "Negotiations, ceasefires, diplomacy"},
            {"id": "16030000", "label": "post-conflict", "definition": "Reconstruction, reconciliation, peacekeeping"}
        ]
    },
    {
        "id": "17000000",
        "label": "weather",
        "definition": "Weather conditions, forecasts, climate patterns",
        "subtopics": [
            {"id": "17010000", "label": "meteorology", "definition": "Weather forecasting, atmospheric science"},
            {"id": "17020000", "label": "extreme weather", "definition": "Storms, floods, heat waves, cold snaps"},
            {"id": "17030000", "label": "climate patterns", "definition": "Seasonal weather, long-term trends"}
        ]
    }
]

# Flatten for embedding
def get_all_iptc_items(categories):
    """Get flattened list of all IPTC items (top-level + subtopics)"""
    items = []
    for cat in categories:
        items.append({
            "id": cat["id"],
            "label": cat["label"],
            "definition": cat["definition"],
            "level": "top",
            "parent_id": None
        })
        for sub in cat.get("subtopics", []):
            items.append({
                "id": sub["id"],
                "label": sub["label"],
                "definition": sub["definition"],
                "level": "sub",
                "parent_id": cat["id"]
            })
    return items

IPTC_ALL_ITEMS = get_all_iptc_items(IPTC_CATEGORIES_V3)

# ============================================================================
# LAYER 1: RULE-BASED MAPPING (EXTENDED FOR SUBTOPICS)
# ============================================================================

# Rule patterns now map to subtopic IDs where possible
RULE_PATTERNS_V3 = {
    # Arts subtopics
    r'^(ART|MUSEUM|PAINTING|SCULPTURE)': '01010000',  # arts
    r'^(CULTURE|HERITAGE|TRADITION)': '01020000',     # culture
    r'^(ENTERTAINMENT|CELEBRITY)': '01030000',        # entertainment
    r'^(MEDIA|PROPAGANDA|CENSOR)': '01040000',       # media

    # Crime subtopics
    r'^(CRIME|KIDNAP|RAPE|DRUG|MURDER|ASSAULT)': '02010000',  # crime
    r'^(ARREST|PRISON|POLICE)': '02020000',                   # law enforcement
    r'^(COURT|TRIAL|VERDICT)': '02030000',                    # justice system

    # Disaster subtopics
    r'^(EARTHQUAKE|TSUNAMI|HURRICANE|FLOOD)': '03010000',     # natural disaster
    r'^(ACCIDENT|EXPLOSION|FIRE)': '03020000',                # man-made disaster
    r'^(EMERGENCY|EVACUATION)': '03030000',                   # emergency response

    # Economy subtopics
    r'^(BUSINESS|CORPORATE|ENTREPRENEUR)': '04010000',        # business
    r'^(BANK|INVEST|STOCK)': '04020000',                      # finance
    r'^(ECONOMY|TRADE|GDP)': '04030000',                      # economy

    # Education subtopics
    r'^(SCHOOL|PRIMARY)': '05010000',                         # primary education
    r'^(UNIVERSITY|COLLEGE)': '05020000',                     # higher education
    r'^(VOCATIONAL|TECHNICAL)': '05030000',                   # vocational training

    # Environment subtopics
    r'^(CLIMATE|EMISSION)': '06010000',                       # climate change
    r'^(CONSERVATION|BIODIVERSITY)': '06020000',              # conservation
    r'^(POLLUTION|CONTAMINATION)': '06030000',                # pollution

    # Health subtopics
    r'^(RESEARCH|DRUG|VACCINE)': '07010000',                  # medical research
    r'^(PUBLIC_HEALTH|EPIDEMIC)': '07020000',                 # public health
    r'^(HOSPITAL|HEALTHCARE)': '07030000',                    # healthcare

    # Religion subtopics
    r'^(CHURCH|MOSQUE|TEMPLE)': '12010000',                   # religious institutions
    r'^(RITUAL|CEREMONY)': '12020000',                        # religious practices
    r'^(FAITH|THEOLOGY)': '12030000',                         # religious beliefs

    # Politics subtopics
    r'^(ELECTION|VOTE)': '11010000',                          # elections
    r'^(GOVERNMENT|POLICY)': '11020000',                      # government
    r'^(DIPLOMACY|TREATY)': '11030000',                       # international relations

    # Conflict subtopics
    r'^(WAR|MILITARY|BATTLE)': '16010000',                    # armed conflict
    r'^(PEACE|NEGOTIATION)': '16020000',                      # peace processes
    r'^(RECONSTRUCTION|PEACEKEEPING)': '16030000',            # post-conflict

    # Sport subtopics
    r'^(PROFESSIONAL|ATHLETE)': '15010000',                   # professional sport
    r'^(OLYMPIC|AMATEUR)': '15020000',                        # amateur sport
    r'^(FEDERATION|ORGANIZATION)': '15030000',                # sports organizations

    # Fallback to top-level (same as v2)
    r'^(ECON|TAX|WB|EPU|FISCAL)': '04000000',
    r'^(POLITICAL|POL|DEMOCRACY)': '11000000',
    r'^(ARMEDCONFLICT|TERROR)': '16000000',
    r'^(CORRUPTION|FRAUD)': '02000000',
    r'^(NATURAL|FAMINE)': '03000000',
    r'^(MEDICAL|DISEASE)': '07000000',
    r'^(ENV|CLIMATE)': '06000000',
    r'^(SOC|POVERTY)': '14000000',
    r'^(RELIGION|CHRISTIAN)': '12000000',
    r'^(SCIENCE|TECH)': '13000000',
    r'^(SPORT|FOOTBALL)': '15000000',
    r'^(TOURISM|TRAVEL)': '10000000',
    r'^(LABOR|EMPLOYMENT)': '09000000',
    r'^(WEATHER|FORECAST)': '17000000',
    r'^(HUMAN_INTEREST)': '08000000',
}

def apply_rule_based_mapping_v3(theme_code: str) -> tuple:
    """
    Apply rule-based mapping to a theme code.
    Returns (iptc_id, confidence) or (None, 0)
    """
    if not theme_code:
        return None, 0

    theme_upper = theme_code.upper()

    for pattern, iptc_id in RULE_PATTERNS_V3.items():
        if re.match(pattern, theme_upper):
            confidence = 0.9 if len(pattern) > 20 else 0.8
            return iptc_id, confidence

    return None, 0

# ============================================================================
# IPTC HIERARCHY FUNCTIONS (New)
# ============================================================================

def load_iptc_hierarchy(json_path: str) -> dict:
    """
    Load IPTC Media Topic hierarchy from JSON file.
    Returns dict mapping qcode to concept info including broader relationships.
    """
    import json

    hierarchy = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for concept in data.get('conceptSet', []):
            qcode = concept.get('qcode', '').replace('medtop:', '')
            if qcode:
                hierarchy[qcode] = {
                    'label': concept.get('prefLabel', {}).get('en-GB', ''),
                    'definition': concept.get('definition', {}).get('en-GB', ''),
                    'broader': [b.replace('medtop:', '') for b in concept.get('broader', [])],
                    'narrower': [n.replace('medtop:', '') for n in concept.get('narrower', [])]
                }
    except Exception as e:
        print(f"[!] Could not load IPTC hierarchy: {e}")

    return hierarchy

def get_top_level_parent(iptc_code: str, hierarchy: dict) -> str:
    """
    Traverse up the hierarchy to find the top-level parent (one of the 17 main categories).
    Returns the top-level IPTC code.
    """
    if not iptc_code or iptc_code not in hierarchy:
        return iptc_code

    current = iptc_code
    visited = set()

    while current and current not in visited:
        visited.add(current)

        # Check if this is a top-level category (no broader or broader is empty)
        concept = hierarchy.get(current, {})
        broader = concept.get('broader', [])

        if not broader:
            # This is a top-level category
            return current

        # Move up to parent
        current = broader[0]  # Take first broader (should be only one)

    # If we can't find a top-level, return original
    return iptc_code

# Global hierarchy cache
IPTC_HIERARCHY = None

def initialize_iptc_hierarchy(data_dir: Path):
    """Initialize IPTC hierarchy from JSON file"""
    global IPTC_HIERARCHY
    if IPTC_HIERARCHY is None:
        json_path = data_dir / "cptall-en-GB.json"
        IPTC_HIERARCHY = load_iptc_hierarchy(str(json_path))
        print(f"  [OK] Loaded IPTC hierarchy with {len(IPTC_HIERARCHY)} concepts")

# ============================================================================
# LAYER 3: SUBTOPIC SIMILARITY SCORING (Modified for hierarchy)
# ============================================================================

def compute_enhanced_similarity(theme_emb, iptc_embs, iptc_items, threshold=0.6):
    """
    Simplified similarity scoring for top-level IPTC categories only.
    All matches are resolved to one of the 17 main IPTC categories.
    """
    sim_matrix = cosine_similarity(theme_emb, iptc_embs)

    results = []
    for i in range(len(sim_matrix)):
        scores = sim_matrix[i]

        # Find best top-level match (all items should be top-level now)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_item = iptc_items[best_idx]

        # Get top-level parent (should be the item itself since we're using top-level only)
        top_level_id = get_top_level_parent(best_item["id"], IPTC_HIERARCHY)

        if best_score >= THR_STRONG_V3:  # Strong match
            results.append({
                "final_id": top_level_id,
                "final_label": best_item["label"],
                "confidence": float(best_score),
                "level": "top-level",
                "top_id": best_item["id"],
                "top_label": best_item["label"],
                "sub_id": None,
                "sub_label": "",
                "sub_definition": ""
            })
        elif best_score >= THR_WEAK_V3:  # Weak match
            results.append({
                "final_id": top_level_id,
                "final_label": best_item["label"],
                "confidence": float(best_score),
                "level": "weak-top-level",
                "top_id": best_item["id"],
                "top_label": best_item["label"],
                "sub_id": None,
                "sub_label": "",
                "sub_definition": ""
            })
        else:
            results.append({
                "final_id": None,
                "final_label": "",
                "confidence": float(best_score),
                "level": "unclassified",
                "top_id": None,
                "top_label": "",
                "sub_id": None,
                "sub_label": "",
                "sub_definition": ""
            })

    return results

# ============================================================================
# FUSION: HIERARCHICAL DECISION TREE
# ============================================================================

THR_STRONG_V3 = 0.25  # Higher threshold for better matches
THR_WEAK_V3 = 0.15    # Higher threshold for fallback matches

def fusion_decision_v3(rule_id: str, rule_conf: float, nn_result: dict, iptc_items: list) -> tuple:
    """
    Simplified fusion for top-level categories only.
    All matches are resolved to one of the 17 main IPTC categories.
    """
    nn_id = nn_result.get("final_id")
    nn_conf = nn_result.get("confidence", 0)
    nn_level = nn_result.get("level", "unclassified")

    # Rule match takes precedence
    if rule_id and rule_conf >= 0.8:
        # Get top-level parent of rule match
        top_level_rule_id = get_top_level_parent(rule_id, IPTC_HIERARCHY)
        return top_level_rule_id, 'rule', rule_conf, {
            "top_id": top_level_rule_id,
            "top_label": get_iptc_label(top_level_rule_id),
            "sub_id": None,
            "sub_label": "",
            "sub_definition": ""
        }

    # NN top-level match
    if nn_level == "top-level" and nn_conf >= THR_STRONG_V3:
        top_level_nn_id = get_top_level_parent(nn_id, IPTC_HIERARCHY)
        return top_level_nn_id, 'nn_top', nn_conf, {
            "top_id": top_level_nn_id,
            "top_label": get_iptc_label(top_level_nn_id),
            "sub_id": None,
            "sub_label": "",
            "sub_definition": ""
        }

    # Weak NN match as fallback
    if nn_level in ["top-level", "weak-top-level"] and nn_conf >= THR_WEAK_V3:
        top_level_nn_id = get_top_level_parent(nn_id, IPTC_HIERARCHY)
        return top_level_nn_id, 'nn_weak', nn_conf, {
            "top_id": top_level_nn_id,
            "top_label": get_iptc_label(top_level_nn_id),
            "sub_id": None,
            "sub_label": "",
            "sub_definition": ""
        }

    return None, 'unclassified', nn_conf, {
        "top_id": None,
        "top_label": "",
        "sub_id": None,
        "sub_label": "",
        "sub_definition": ""
    }

# ============================================================================
# MAIN PIPELINE (Modified for v3)
# ============================================================================

def run_full_mapping_v3(data_dir: str = ".", output_dir: str = ".", mapping_source: str = 'vargo') -> dict:
    """
    Run the complete three-layer GDELT -> IPTC mapping pipeline with top-level categories only.

    Returns:
        Dictionary with mapping results and statistics
    """
    print("=" * 60)
    print("GDELT Theme -> IPTC Media Topics Mapper (v3)")
    print("Simplified Approach: Top-level Categories Only")
    print("=" * 60)

    output_path = Path(output_dir)
    data_path = Path(data_dir)
    results_path = Path(output_dir)  # Results are saved to output_dir

    # Initialize IPTC hierarchy
    print("\n[*] Loading IPTC Media Topic hierarchy...")
    initialize_iptc_hierarchy(data_path)

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
            df = pd.read_csv(results_path / csv_file)
            if 'theme_code' in df.columns:
                theme_codes.update(df['theme_code'].dropna().unique())
        except Exception as e:
            print(f"  [!] Could not read {csv_file}: {e}")

    if not theme_codes:
        print("[X] No theme codes found in CSV files!")
        return None

    print(f"  [OK] Found {len(theme_codes)} unique theme codes")

    # Load mapping source
    print(f"\n[*] Loading {mapping_source.upper()} mapping...")
    mapping_data = load_mapping_source(data_path, mapping_source)
    print(f"  [OK] Loaded {len(mapping_data)} {mapping_source.upper()} mappings")

    # LAYER 1: Rule-based mapping (extended for subtopics)
    print("\n[*] Layer 1: Applying rule-based mapping...")
    rule_results = {}
    rule_matched = 0

    for theme in theme_codes:
        iptc_id, confidence = apply_rule_based_mapping_v3(theme)
        rule_results[theme] = {'rule_id': iptc_id, 'rule_conf': confidence}
        if iptc_id:
            rule_matched += 1

    print(f"  [OK] Rule-based matched: {rule_matched}/{len(theme_codes)} ({100*rule_matched/len(theme_codes):.1f}%)")

    # LAYER 2 & 3: Embedding-based mapping with top-level categories only
    if not TRANSFORMERS_AVAILABLE:
        print("\n[!] Sentence-transformers not available. Using rule-based only.")
        nn_results = {theme: {
            'nn_id': None, 'nn_label': '', 'nn_score': 0,
            'level': 'unclassified', 'top_id': None, 'sub_id': None
        } for theme in theme_codes}
    else:
        print("\n[*] Layer 2-3: Computing semantic embeddings with top-level categories only...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        print("  Loading sentence-transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Back to proven model

        # Build text representations - only top-level categories for better matching
        theme_list = sorted(theme_codes)
        theme_texts = [build_theme_text(t, mapping_data, mapping_source) for t in theme_list]
        
        # Only encode top-level IPTC categories (17 main categories)
        top_level_items = [item for item in IPTC_ALL_ITEMS if item["level"] == "top"]
        iptc_texts = [item['definition'] for item in top_level_items]

        print(f"  Encoding {len(theme_texts)} themes...")
        theme_embeddings = compute_embeddings(theme_texts, model)

        print(f"  Encoding {len(iptc_texts)} IPTC top-level categories...")
        iptc_embeddings = compute_embeddings(iptc_texts, model)

        print("  Computing similarities...")
        nn_list = compute_enhanced_similarity(theme_embeddings, iptc_embeddings, top_level_items)

        # Debug: Print similarity statistics
        confidences = [r['confidence'] for r in nn_list]
        print(f"  Similarity stats: min={min(confidences):.3f}, max={max(confidences):.3f}, mean={np.mean(confidences):.3f}, median={np.median(confidences):.3f}")
        print(f"  Thresholds: strong={THR_STRONG_V3}, weak={THR_WEAK_V3}")

        nn_results = {theme: nn_list[i] for i, theme in enumerate(theme_list)}

        # Compute 2D projections for visualization
        print("\n[*] Computing 2D projections for visualization...")
        coords_2d = compute_2d_projection(theme_embeddings, method='tsne')

        # Store coordinates in nn_results
        for i, theme in enumerate(theme_list):
            nn_results[theme]['tsne_x'] = float(coords_2d[i, 0])
            nn_results[theme]['tsne_y'] = float(coords_2d[i, 1])

    # FUSION
    print("\n[*] Fusion: Combining rule-based and hierarchical embedding results...")

    final_results = []
    decision_counts = {}

    for theme in sorted(theme_codes):
        rule_id = rule_results[theme]['rule_id']
        rule_conf = rule_results[theme]['rule_conf']

        nn_data = nn_results.get(theme, {
            'final_id': None, 'final_label': '', 'confidence': 0,
            'level': 'unclassified', 'top_id': None, 'sub_id': None
        })

        final_id, decision, confidence, subtopic_info = fusion_decision_v3(
            rule_id, rule_conf, nn_data, IPTC_ALL_ITEMS
        )

        decision_counts[decision] = decision_counts.get(decision, 0) + 1

        result = {
            'theme_code': theme,
            'text_repr': build_theme_text(theme, mapping_data, mapping_source),
            'iptc_rule_id': rule_id,
            'iptc_rule_label': get_iptc_label(rule_id) if rule_id else '',
            'rule_confidence': rule_conf,
            'iptc_nn_id': nn_data.get('final_id'),
            'iptc_nn_label': nn_data.get('final_label', ''),
            'nn_score': nn_data.get('confidence', 0),
            'iptc_final_id': final_id,
            'iptc_final_label': get_iptc_label(final_id) if final_id else '',
            'decision_source': decision,
            'final_confidence': confidence,
            # Subtopic information
            'subtopic_info': subtopic_info,
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

    # Category distribution (top-level only for compatibility)
    category_dist = {}
    for r in final_results:
        cat = r['iptc_final_label'] or 'unclassified'
        category_dist[cat] = category_dist.get(cat, 0) + 1

    print("\n[*] IPTC Category Distribution (Top-level):")
    for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save results
    print("\n[*] Saving results...")

    # Build output structure (maintain v2 compatibility)
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_themes": len(theme_codes),
            "rule_matched": rule_matched,
            "iptc_categories": len(set(r['iptc_final_label'] for r in final_results if r['iptc_final_label'])),
            "decision_stats": decision_counts,
            "mapping_source": mapping_source,
            "version": "v3",
            "subtopics_enabled": False
        },
        "themes": final_results,
        "iptc_categories": {}
    }

    # Group by IPTC category (top-level for compatibility)
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

    # Clean data for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output_data = clean_for_json(output_data)

    # Save JSON
    json_filename = f"gdelt_iptc_mapping_v3_{mapping_source}.json"
    json_path = output_path / json_filename
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  [OK] JSON: {json_path.name}")

    # Save CSV
    csv_filename = f"gdelt_themes_iptc_v3_{mapping_source}.csv"
    csv_path = output_path / csv_filename
    pd.DataFrame(final_results).to_csv(csv_path, index=False)
    print(f"  [OK] CSV: {csv_path.name}")

    print("\n[OK] V3 Mapping complete!")
    print(f"\n[*] Summary:")
    print(f"   Total themes: {len(theme_codes)}")
    print(f"   Top-level matches: {decision_counts.get('nn_top', 0) + decision_counts.get('nn_weak', 0) + decision_counts.get('rule', 0)}")
    print(f"   Unclassified: {decision_counts.get('unclassified', 0)}")

    return output_data

# ============================================================================
# UTILITY FUNCTIONS (from v2)
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


def load_gkg_mapping(filepath: str = "gdelt_gkg_categorylist.csv") -> dict:
    """Load GKG Category List mapping - filtered to Theme entries only"""
    gkg_map = {}
    try:
        df = pd.read_csv(filepath)
        # Filter to only Theme entries
        theme_df = df[df['Type'] == 'Theme']
        for _, row in theme_df.iterrows():
            gkg_map[row['Name']] = {
                'description': row.get('Description', '')
            }
    except Exception as e:
        print(f"[!] Could not load GKG mapping: {e}")
    return gkg_map


def load_combined_mapping(data_dir: Path) -> dict:
    """Load combined Vargo + GKG mapping for richer descriptions"""
    combined_map = {}

    # Load Vargo mapping
    vargo_map = load_vargo_mapping(data_dir / "vargo_gdelt_themes_issues.csv")

    # Load GKG mapping
    gkg_map = load_gkg_mapping(data_dir / "gdelt_gkg_categorylist.csv")

    # Combine mappings
    all_theme_codes = set(vargo_map.keys()) | set(gkg_map.keys())

    for theme_code in all_theme_codes:
        descriptions = []

        # Add GKG description if available
        if theme_code in gkg_map and gkg_map[theme_code].get('description'):
            descriptions.append(gkg_map[theme_code]['description'])

        # Add Vargo description if available
        if theme_code in vargo_map:
            vargo_info = vargo_map[theme_code]
            if vargo_info.get('description'):
                descriptions.append(vargo_info['description'])
            if vargo_info.get('issue'):
                descriptions.append(f"Issue: {vargo_info['issue']}")

        # Combine all descriptions
        combined_description = ' | '.join(descriptions) if descriptions else ''

        combined_map[theme_code] = {
            'description': combined_description,
            'sources': []
        }

        if theme_code in gkg_map:
            combined_map[theme_code]['sources'].append('gkg')
        if theme_code in vargo_map:
            combined_map[theme_code]['sources'].append('vargo')

    print(f"  [OK] Combined {len(combined_map)} mappings from Vargo ({len(vargo_map)}) + GKG ({len(gkg_map)})")

    return combined_map


def load_mapping_source(data_dir: Path, mapping_source: str) -> dict:
    """Load mapping source based on selection."""
    if mapping_source == 'vargo':
        return load_vargo_mapping(data_dir / "vargo_gdelt_themes_issues.csv")
    elif mapping_source == 'gkg':
        return load_gkg_mapping(data_dir / "gdelt_gkg_categorylist.csv")
    elif mapping_source == 'combined':
        return load_combined_mapping(data_dir)
    else:
        print(f"[!] Unknown mapping source: {mapping_source}, using combined as default")
        return load_combined_mapping(data_dir)


def build_theme_text(theme_code: str, mapping_data: dict, mapping_source: str) -> str:
    """Build text representation using only descriptions for semantic comparison"""
    if theme_code in mapping_data:
        info = mapping_data[theme_code]
        description = info.get('description', '')
        if description:
            return description
    # Fallback if no description
    return theme_code.replace('_', ' ').replace('-', ' ').lower()


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


def get_iptc_label(iptc_id: str) -> str:
    """Get IPTC category label from ID"""
    for cat in IPTC_CATEGORIES_V3:
        if cat['id'] == iptc_id:
            return cat['label']
        for sub in cat.get('subtopics', []):
            if sub['id'] == iptc_id:
                return f"{cat['label']} - {sub['label']}"
    return ''

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Map GDELT themes to IPTC Media Topics (V3 - Top-level Categories Only)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--mapping-source",
        type=str,
        choices=['vargo', 'gkg', 'combined'],
        default='combined',
        help="Mapping source to use: 'vargo' (Vargo issue taxonomy), 'gkg' (GKG Category List), or 'combined' (both)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"

    results = run_full_mapping_v3(data_dir=data_dir, output_dir=output_dir, mapping_source=args.mapping_source)