# GDELT Theme → IPTC Media Topics Mapper

Semantic embedding-based classification of GDELT news themes into IPTC Media Topics categories using Sentence-Transformers with dual algorithm support.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GDELT](https://img.shields.io/badge/Data-GDELT%20GKG-orange.svg)](https://www.gdeltproject.org/)
[![IPTC](https://img.shields.io/badge/Taxonomy-IPTC%20Media%20Topics-purple.svg)](https://iptc.org/standards/media-topics/)

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Data Sources](#-data-sources)
- [Dashboard](#-dashboard)
- [API Reference](#-api-reference)
- [References](#-references)
- [License](#-license)

---

## Overview

This project maps **GDELT Global Knowledge Graph (GKG) themes** to the standardized **IPTC Media Topics** taxonomy using semantic embeddings. The mapping enables cross-platform news categorization and comparative media analysis across different countries.

### Key Objectives

1. **Standardize** GDELT's proprietary theme codes to industry-standard IPTC categories
2. **Enable** cross-country comparative analysis of news coverage
3. **Provide** an interactive dashboard for exploring theme distributions
4. **Compare** two different mapping algorithms (V1 vs V2)
5. **Export** results in multiple formats (CSV, XLSX, JSON, LaTeX, TikZ)

---

## Features

### Dual Algorithm Support

#### V1: Embedding-Only Approach

- Pure semantic similarity using Sentence-Transformers
- Direct cosine similarity matching
- Best for themes with clear semantic meaning

#### V2: Two-Layer Fusion Approach

- **Layer 1**: Rule-based keyword matching for common patterns
- **Layer 2**: Embedding-based NN for remaining themes
- Higher accuracy through combined approach
- Configurable fusion weights

### Interactive Dashboard

- Real-time data visualization with Chart.js
- **Side-by-side t-SNE scatter plots** for V1 vs V2 comparison
- Country and theme filtering (6 countries)
- IPTC category color-coded displays
- Three data tables (Total Docs, Monthly Quality, Monthly Detail)
- Responsive design

### Visualization Charts

- t-SNE 2D theme distribution (real coordinates from sklearn)
- IPTC category pie/doughnut charts
- Theme count by category
- Similarity score distribution
- Country-wise document volume

### Export Capabilities

- XLSX (multi-sheet workbooks)
- CSV (comma-separated values)
- JSON (structured data)
- LaTeX (academic tables)
- TikZ (LaTeX treemaps for publications)
- PNG/SVG (chart graphics)

### Analysis Tools

- Document volume analysis by country
- Monthly quality metrics
- Theme distribution statistics
- IPTC category summaries
- Algorithm comparison metrics
- **Treemap generation** for IPTC categories (LaTeX TikZ)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/omeralperguzel/GDELT-IPTC-Mapper.git
cd GDELT-IPTC-Mapper
```

2. **Install dependencies**

```bash
pip install pandas numpy sentence-transformers scikit-learn squarify matplotlib
```

3. **Start the server**

```bash
python run_server.py
```

4. **Open the dashboard**

```
http://localhost:5001
```

---

## Usage

### 1. Load Data

Navigate to the **GDELT Tema** tab and click **" CSV Verilerini Yükle"** to load the BigQuery CSV files:

- Table 1: Total document counts by country-theme
- Table 2: Monthly quality metrics
- Table 3: Monthly detail data

### 2. Run IPTC Mapping

Go to the **Kümeleme** tab and choose an algorithm:

| Button          | Algorithm        | Description              |
| --------------- | ---------------- | ------------------------ |
| **V1 Çalıştır** | Embedding Only   | Pure semantic similarity |
| **V2 Çalıştır** | Two-Layer Fusion | Rule + Embedding hybrid  |

Or load existing results with **V1 Yükle** / **V2 Yükle** buttons.

### 3. Switch Active Results

Use the dropdown menu to select which algorithm's results to use for theme analysis:

- V1 results for embedding-based grouping
- V2 results for rule+embedding fusion grouping

### 4. Explore Results

- View theme-IPTC mappings in the results table
- Check similarity scores and confidence levels
- Filter by country or IPTC category
- Compare V1 vs V2 assignments

### 5. Generate Charts

Switch to the **Grafikler** tab to visualize:

- **t-SNE 2D Scatter Plots** (V1 and V2 side-by-side)
- IPTC category distribution
- Country-wise theme volumes
- Top themes by document count
- Similarity score analysis

### 6. Generate Treemaps

Use the **Treemap** tab to generate LaTeX TikZ treemaps:

- **Single Category**: Generate treemap for specific IPTC category
- **All Categories**: Generate treemaps for all 17 IPTC categories
- Output files saved in `latex_treemaps/` directory

### 7. Export Data

Use the **Dışarı Aktar** tab to download:

- Filtered data tables (XLSX/CSV)
- IPTC mapping results (JSON)
- Chart graphics (PNG/SVG)
- Complete reports (ZIP)

---

## Project Structure

```
GDELT-IPTC-Mapper/
│
├──  Core Application
│   ├── index.html                    # Main dashboard (HTML/CSS/JS)
│   ├── run_server.py                 # Python HTTP server with API
│   └── analysis.py                   # Data analysis utilities
│
├──  Mapping Algorithms
│   ├── gdelt_iptc_mapping.py         # V1: Embedding-only pipeline
│   ├── gdelt_iptc_mapping_v2.py      # V2: Two-layer fusion pipeline
│   └── gdelt_theme_clustering.py     # Alternative clustering approach
│
├──  Data Files (BigQuery Exports)
│   ├── bquxjob_645c6baa_*.csv        # Total docs per country-theme
│   ├── bquxjob_4750d984_*.csv        # Monthly quality metrics
│   └── bquxjob_5c135702_*.csv        # Monthly detail data
│
├──  Mapping Reference Files
│   ├── vargo_gdelt_themes_issues.csv # GDELT theme → Issue mapping
│   └── iptc_mediatopics.csv          # IPTC taxonomy (17 categories)
│
├──  Output Files
│   ├── gdelt_iptc_mapping_v1.json    # V1 mapping results
│   ├── gdelt_iptc_mapping_v2.json    # V2 mapping results
│   ├── gdelt_themes_iptc_v1.csv      # V1 theme-IPTC pairs
│   ├── gdelt_themes_iptc_v2.csv      # V2 theme-IPTC pairs
│   └── gdelt_theme_clusters.json     # Clustering results
│
├──  Saved Analyses
│   └── saved_analyses/               # Auto-saved analysis states
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Methodology

### 1. Data Collection

GDELT GKG data is queried from Google BigQuery for 6 countries (2022-2024):

| Code | Country      | Flag |
| ---- | ------------ | ---- |
| CE   | Sri Lanka    |      |
| HO   | Honduras     |      |
| HR   | Croatia      |      |
| KG   | Kyrgyzstan   |      |
| LO   | Slovakia     |      |
| SA   | Saudi Arabia |      |

### 2. Algorithm Comparison

#### V1: Embedding-Only Pipeline

```
GDELT Theme → Text Representation → Sentence-BERT → Cosine Similarity → IPTC Match
```

1. Convert theme code to descriptive text
2. Generate 384-dim embedding with all-MiniLM-L6-v2
3. Compute cosine similarity with all 17 IPTC category embeddings
4. Assign to highest similarity category

#### V2: Two-Layer Fusion Pipeline

```
GDELT Theme → [Rule Check] → Match? → Use Rule Result
                    ↓ No
              [Embedding NN] → IPTC Match
```

1. **Layer 1 (Rules)**: Check keyword patterns for common themes

   - TAX*\*, ECON*\* → economy, business and finance
   - EPU*\*, GOV*\* → politics and government
   - HEALTH*\*, DISEASE*\* → health
   - etc.

2. **Layer 2 (Embedding)**: For unmatched themes, use semantic similarity

3. **Fusion**: Combine results with rule priority

### 3. t-SNE Visualization

Real 2D coordinates are computed using sklearn:

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# PCA pre-reduction: 384 → 50 dimensions
pca = PCA(n_components=50)
reduced = pca.fit_transform(embeddings)

# t-SNE: 50 → 2 dimensions
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
coords_2d = tsne.fit_transform(reduced)
```

### 4. Similarity Calculation

Cosine similarity between theme and IPTC embeddings:

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

### 5. Treemap Generation

LaTeX TikZ treemaps are generated using squarify algorithm:

```python
import squarify

# Generate rectangles for treemap
sizes = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(sizes, 0, 0, width, height)
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        GDELT GKG Themes                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   V1: Embedding Only    │     │   V2: Two-Layer Fusion  │
├─────────────────────────┤     ├─────────────────────────┤
│ • Text Representation   │     │ • Rule-based matching   │
│ • Sentence-BERT encode  │     │ • Keyword patterns      │
│ • Cosine similarity     │     │ • Embedding fallback    │
│ • Direct NN assignment  │     │ • Combined scoring      │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ gdelt_iptc_mapping_v1   │     │ gdelt_iptc_mapping_v2   │
│        .json            │     │        .json            │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
              ┌─────────────────────────┐
              │    Interactive Dashboard │
              │  • Side-by-side t-SNE   │
              │  • Algorithm comparison │
              │  • Export capabilities  │
              │  • Treemap generation   │
              └─────────────────────────┘
```

---

## Data Sources

### GDELT Global Knowledge Graph

- **Source**: [GDELT Project](https://www.gdeltproject.org/)
- **Access**: Google BigQuery (`gdelt-bq.gdeltv2.gkg`)
- **Period**: 2022-2024
- **Countries**: 6 (CE, HO, HR, KG, LO, SA)

### IPTC Media Topics

- **Source**: [IPTC NewsCodes](https://iptc.org/standards/media-topics/)
- **Version**: Media Topics 2024
- **Categories**: 17 top-level categories

| ID  | IPTC Category                             |
| --- | ----------------------------------------- |
| 01  | arts, culture, entertainment and media    |
| 02  | conflict, war and peace                   |
| 03  | crime, law and justice                    |
| 04  | disaster, accident and emergency incident |
| 05  | economy, business and finance             |
| 06  | education                                 |
| 07  | environment                               |
| 08  | health                                    |
| 09  | human interest                            |
| 10  | labour                                    |
| 11  | lifestyle and leisure                     |
| 12  | politics and government                   |
| 13  | religion                                  |
| 14  | science and technology                    |
| 15  | society                                   |
| 16  | sport                                     |
| 17  | weather                                   |

---

## Dashboard

### Tab Overview

| Tab                | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| **GDELT Tema**     | Load CSV data, view 3 data tables, filter by 6 countries, run theme analysis |
| **Grafikler**      | Interactive charts including side-by-side t-SNE V1/V2 scatter plots          |
| **Kümeleme**       | Run V1 or V2 algorithm, load results, switch active mapping                  |
| � **Treemap**      | Generate LaTeX TikZ treemaps for IPTC categories                             |
| � **Dışarı Aktar** | Export data in XLSX, CSV, JSON, LaTeX formats                                |

### Kümeleme Tab Features

- **Dual Algorithm Buttons**: Run V1 or V2 independently
- **Status Indicators**: Show loaded theme counts for each version
- **Active Mapping Selector**: Choose which results to use for analysis
- **Progress Tracker**: 6-step pipeline visualization

### Charts Tab Features

- **t-SNE V1 Scatter**: 2D projection of V1 embeddings
- **t-SNE V2 Scatter**: 2D projection of V2 embeddings
- **IPTC Category Distribution**: Doughnut chart
- **Theme Count by Category**: Bar chart
- **Similarity Score Distribution**: Histogram

### Treemap Tab Features

- **Single IPTC Category**: Generate TikZ treemap for one category
- **All IPTC Categories**: Batch generate treemaps for all 17 categories
- **LaTeX Output**: Files saved in `latex_treemaps/` directory
- **Publication Ready**: High-quality vector graphics for academic papers

---

## API Reference

### Endpoints

| Method | Endpoint                          | Description                  |
| ------ | --------------------------------- | ---------------------------- |
| GET    | `/`                               | Serve main dashboard         |
| GET    | `/*.csv`                          | Serve CSV data files         |
| GET    | `/*.json`                         | Serve JSON results           |
| POST   | `/api/analyze`                    | Run data analysis            |
| POST   | `/api/run-iptc-mapping`           | Execute mapping pipeline     |
| POST   | `/api/generate-iptc-treemap`      | Generate single IPTC treemap |
| POST   | `/api/generate-all-iptc-treemaps` | Generate all IPTC treemaps   |
| POST   | `/api/save`                       | Save analysis state          |
| POST   | `/api/load`                       | Load saved analysis          |
| POST   | `/api/save-analysis`              | Save analysis with filename  |
| GET    | `/api/load-analysis/{filename}`   | Load specific saved analysis |
| GET    | `/api/list-saved-analyses`        | List all saved analyses      |
| DELETE | `/api/delete-analysis/{filename}` | Delete saved analysis        |

### IPTC Mapping API

```bash
# Run V1 algorithm (embedding only)
curl -X POST http://localhost:5001/api/run-iptc-mapping \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "v1"}'

# Run V2 algorithm (two-layer fusion)
curl -X POST http://localhost:5001/api/run-iptc-mapping \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "v2"}'
```

### Treemap Generation API

```bash
# Generate treemap for single IPTC category
curl -X POST http://localhost:5001/api/generate-iptc-treemap \
  -H "Content-Type: application/json" \
  -d '{"iptc_category": "economy, business and finance"}'

# Generate treemaps for all categories
curl -X POST http://localhost:5001/api/generate-all-iptc-treemaps
```

### Response Format

```json
{
  "metadata": {
    "total_themes": 285,
    "iptc_categories": 17,
    "algorithm": "v2",
    "algorithm_name": "Two-Layer Fusion"
  },
  "themes": [
    {
      "theme_code": "ECON_BANKRUPTCY",
      "iptc_final_label": "economy, business and finance",
      "nn_score": 0.72,
      "tsne_x": 45.2,
      "tsne_y": 32.8
    }
  ],
  "iptc_categories": {
    "economy, business and finance": {
      "theme_count": 45,
      "themes": ["ECON_BANKRUPTCY", "TAX_FNCACT", ...]
    }
  }
}
```

---

## References

### Academic Papers

1. **Tarekegn, A. N., et al. (2024)**. "GDELT-based analysis using LLM clustering". _Journal of Computational Social Science_.

2. **Kuzman, T., & Ljubešić, N. (2024)**. "News categorization using IPTC Media Topics". _Proceedings of LREC-COLING 2024_.

3. **Vargo, C. J., & Guo, L. (2017)**. "Networks, Big Data, and Intermedia Agenda Setting". _Journalism & Mass Communication Quarterly_.

4. **Reimers, N., & Gurevych, I. (2019)**. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". _EMNLP 2019_.

5. **van der Maaten, L., & Hinton, G. (2008)**. "Visualizing Data using t-SNE". _Journal of Machine Learning Research_.

### Data Sources

- [GDELT Project](https://www.gdeltproject.org/)
- [IPTC Media Topics](https://iptc.org/standards/media-topics/)
- [Sentence-Transformers](https://www.sbert.net/)

---

## Author

**Ömer Alper Güzel**  
TED University, Department of Computer Engineering  
 omer.guzel@tedu.edu.tr

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- GDELT Project for providing open access to global news data
- IPTC for the standardized Media Topics taxonomy
- Hugging Face for the Sentence-Transformers library
- scikit-learn for t-SNE implementation
- squarify for treemap generation
- TED University CMPE490 course

---

<p align="center">
  Made with  for media analysis research
</p>
