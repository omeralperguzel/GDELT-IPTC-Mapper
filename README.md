# 🌐 GDELT Theme → IPTC Media Topics Mapper

Semantic embedding-based classification of GDELT news themes into IPTC Media Topics categories using Sentence-Transformers.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GDELT](https://img.shields.io/badge/Data-GDELT%20GKG-orange.svg)](https://www.gdeltproject.org/)
[![IPTC](https://img.shields.io/badge/Taxonomy-IPTC%20Media%20Topics-purple.svg)](https://iptc.org/standards/media-topics/)

---

## 📋 Table of Contents

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

## 🎯 Overview

This project maps **GDELT Global Knowledge Graph (GKG) themes** to the standardized **IPTC Media Topics** taxonomy using semantic embeddings. The mapping enables cross-platform news categorization and comparative media analysis across different countries.

### Key Objectives

1. **Standardize** GDELT's proprietary theme codes to industry-standard IPTC categories
2. **Enable** cross-country comparative analysis of news coverage
3. **Provide** an interactive dashboard for exploring theme distributions
4. **Export** results in multiple formats (CSV, XLSX, JSON, LaTeX)

---

## ✨ Features

### 🔬 Semantic Mapping

- Sentence-Transformer embeddings (all-MiniLM-L6-v2)
- Cosine similarity-based nearest neighbor assignment
- Confidence scores and second-best matches
- Support for 17 IPTC top-level categories

### 📊 Interactive Dashboard

- Real-time data visualization with Chart.js
- Country and theme filtering
- IPTC category color-coded displays
- Responsive design

### 📤 Export Capabilities

- XLSX (multi-sheet workbooks)
- CSV (comma-separated values)
- JSON (structured data)
- LaTeX (academic tables)
- PNG/SVG (chart graphics)

### 🔧 Analysis Tools

- Document volume analysis
- Monthly quality metrics
- Theme distribution statistics
- IPTC category summaries

---

## 🚀 Installation

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
pip install pandas numpy sentence-transformers scikit-learn
```

3. **Start the server**

```bash
python run_server.py
```

4. **Open the dashboard**

```
http://localhost:5000
```

---

## 📖 Usage

### 1. Load Data

Navigate to the **GDELT Tema** tab and click **"📥 CSV Verilerini Yükle"** to load the BigQuery CSV files.

### 2. Run IPTC Mapping

Go to the **Kümeleme** tab and click **"🚀 IPTC Eşleştirme Çalıştır"** to generate semantic mappings.

### 3. Explore Results

- View theme-IPTC mappings in the results table
- Check similarity scores and confidence levels
- Filter by country or IPTC category

### 4. Generate Charts

Switch to the **Grafikler** tab and click **"🔄 Grafikleri Oluştur"** to visualize:

- IPTC category distribution
- Country-wise theme volumes
- Top themes by document count
- Similarity score analysis

### 5. Export Data

Use the **Dışarı Aktar** tab to download:

- Filtered data tables (XLSX/CSV)
- IPTC mapping results (JSON)
- Chart graphics (PNG/SVG)
- Complete reports (ZIP)

---

## 📁 Project Structure

```
GDELT-IPTC-Mapper/
│
├── index.html                    # Main dashboard (HTML/CSS/JS)
├── run_server.py                 # Python HTTP server with API
├── gdelt_iptc_mapping.py         # Semantic mapping pipeline
├── analysis.py                   # Data analysis utilities
│
├── 📊 Data Files
│   ├── bquxjob_645c6baa_*.csv   # Total docs per country-theme
│   ├── bquxjob_4750d984_*.csv   # Monthly quality metrics
│   └── bquxjob_5c135702_*.csv   # Monthly detail data
│
├── 📋 Mapping Files
│   ├── vargo_gdelt_themes_issues.csv  # GDELT theme → Issue mapping
│   ├── iptc_mediatopics.csv           # IPTC taxonomy (17 categories)
│   ├── gdelt_iptc_mapping.json        # Mapping results
│   └── gdelt_themes_iptc.csv          # Theme-IPTC pairs
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔬 Methodology

### 1. Data Collection

GDELT GKG data is queried from Google BigQuery for 6 countries (2022-2024):

- 🇱🇰 Sri Lanka (CE)
- 🇭🇳 Honduras (HO)
- 🇭🇷 Croatia (HR)
- 🇰🇬 Kyrgyzstan (KG)
- 🇸🇰 Slovakia (LO)
- 🇸🇦 Saudi Arabia (SA)

### 2. Theme Representation

Each GDELT theme is converted to a text representation:

```python
"ECON - Economy / Finance: Economic events and financial news"
```

### 3. Embedding Generation

Text representations are encoded using Sentence-BERT:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(theme_texts)
```

### 4. Similarity Calculation

Cosine similarity is computed between each theme and all 17 IPTC categories:

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

### 5. Category Assignment

Each theme is assigned to its nearest IPTC category:

```python
best_match = argmax(cosine_similarity(theme_embedding, iptc_embeddings))
```

### Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GDELT GKG     │────▶│  Text Repr.     │────▶│  Sentence-BERT  │
│   (18 themes)   │     │  Generation     │     │  Encoding       │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │◀────│  JSON/CSV       │◀────│  Cosine Sim.    │
│   (HTML/JS)     │     │  Results        │     │  Matching       │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  IPTC Topics    │
                                                │  (17 categories)│
                                                └─────────────────┘
```

---

## 📚 Data Sources

### GDELT Global Knowledge Graph

- **Source**: [GDELT Project](https://www.gdeltproject.org/)
- **Access**: Google BigQuery (`gdelt-bq.gdeltv2.gkg`)
- **Period**: 2022-2024
- **Themes**: 18 unique theme codes

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

## 🖥️ Dashboard

### Tab Overview

| Tab                 | Description                                            |
| ------------------- | ------------------------------------------------------ |
| 📰 **GDELT Tema**   | Load CSV data, view theme tables, filter by country    |
| 📈 **Grafikler**    | Interactive charts (bar, horizontal bar, distribution) |
| 🤖 **Kümeleme**     | Run IPTC mapping, view similarity scores               |
| 📤 **Dışarı Aktar** | Export data in multiple formats                        |

### Screenshots

The dashboard provides:

- Dark navy theme with modern UI
- Color-coded IPTC categories
- Sortable and filterable tables
- Responsive chart visualizations

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint                | Description              |
| ------ | ----------------------- | ------------------------ |
| GET    | `/`                     | Serve main dashboard     |
| GET    | `/*.csv`                | Serve CSV data files     |
| GET    | `/*.json`               | Serve JSON results       |
| POST   | `/api/run-iptc-mapping` | Execute mapping pipeline |

### Example Request

```bash
curl -X POST http://localhost:5000/api/run-iptc-mapping \
  -H "Content-Type: application/json"
```

---

## 📖 References

### Academic Papers

1. **Tarekegn, A. N., et al. (2024)**. "GDELT-based analysis using LLM clustering". _Journal of Computational Social Science_.

2. **Kuzman, T., & Ljubešić, N. (2024)**. "News categorization using IPTC Media Topics". _Proceedings of LREC-COLING 2024_.

3. **Vargo, C. J., & Guo, L. (2017)**. "Networks, Big Data, and Intermedia Agenda Setting". _Journalism & Mass Communication Quarterly_.

4. **Reimers, N., & Gurevych, I. (2019)**. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". _EMNLP 2019_.

### Data Sources

- [GDELT Project](https://www.gdeltproject.org/)
- [IPTC Media Topics](https://iptc.org/standards/media-topics/)
- [Sentence-Transformers](https://www.sbert.net/)

---

## 👤 Author

**Ömer Alper Güzel**  
TED University, Department of Computer Engineering  
📧 omer.guzel@tedu.edu.tr

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- GDELT Project for providing open access to global news data
- IPTC for the standardized Media Topics taxonomy
- Hugging Face for the Sentence-Transformers library
- TED University CMPE490 course

---

<p align="center">
  Made with ❤️ for media analysis research
</p>
