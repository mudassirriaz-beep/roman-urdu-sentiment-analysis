# Roman Urdu Sentiment Analysis

## Overview
This project provides a **sentiment analysis model** for **Roman Urdu + English code-mixed text** (e.g., `"yeh movie bohat achi thi"`). It classifies text into **positive, negative, or neutral** using a fine‑tuned XLM‑RoBERTa model.

## Data Sources & Preparation
- **Kaggle UCI Roman Urdu Dataset** – 20,000+ labeled sentences (positive/negative/neutral).  
- **Self‑collected / augmented data** – 1,800+ synthetic examples targeting weak patterns (negation, mixed sentiment, sarcasm).

### Data Cleaning
- Removed non‑text rows (numbers, symbols, empty strings)
- Normalized Roman Urdu spellings (`accha` → `acha`, `bohot` → `bohat`)
- Standardised labels: `positive` (2), `neutral` (1), `negative` (0)

### Data Augmentation
- Synonym replacement for minority classes
- Targeted generation of neutral‑negation phrases (`"kuch khaas nahi"`)
- Mixed sentiment examples (`"pehle acha phir bura"`)
- Emoji to text conversion (`😂` → `laughing`)

Final cleaned dataset: **~20,000 rows** (original) + **~1,800 synthetic** → balanced across three classes.

## Model Training & Improvement
- **Base model**: `xlm-roberta-base` (Hugging Face)
- **Training**: 8 epochs, class weights, early stopping, low learning rate (2e‑5)
- **Post‑processing rules** to fix stubborn edge cases (e.g., `"kamaal kar diya"` → positive)

### Performance Metrics
| Test Set | Accuracy |
|----------|----------|
| Held‑out test split (20% of original) | **96%** |
| 40 brand‑new, unseen real‑world sentences | **83%** |

**Confusion matrix (40 sentences):**
- Positive: precision 0.85, recall 0.94
- Negative: precision 0.80, recall 0.75
- Neutral: precision 0.82, recall 0.78

## Features
- 🧠 Fine‑tuned **XLM‑RoBERTa** model
- 🚀 **FastAPI** web interface with modern CSS (gradient, responsive cards)
- 📂 Bulk upload (CSV/TXT) with **Matplotlib bar chart** showing sentiment distribution
- 💾 **Portable Windows app** – USB‑ready, no Python installation needed (uses WinPython)

## How to Use (Portable USB Version)
1. Download **WinPython 64‑bit (dot version)** from [winpython.github.io](https://winpython.github.io/).
2. Extract to a folder (e.g., `D:\sentiment analysis`).
3. Copy the following files into that folder:
   - `app.py`
   - `final_model_v6` (entire folder) – download from [Google Drive link] (add your link)
   - `requirements.txt`
   - `start_app.bat`
   - `install_deps.bat`
4. **First time only:** double‑click `install_deps.bat` (installs all dependencies, ~5‑10 min).
5. **Every time:** double‑click `start_app.bat`. Browser opens at `http://localhost:8000`.
6. Enter a sentence or upload a CSV/TXT file → get sentiment + confidence + chart.

## Tech Stack
- **Transformers** (Hugging Face) – XLM‑RoBERTa
- **PyTorch** (CUDA 12.8 / CPU fallback)
- **FastAPI** + Uvicorn
- **Matplotlib, Pandas**
- **WinPython** (portable distribution)

## File Structure
sentiment_analysis/
├── final_model_v6/ # Trained model (1.5 GB) – download separately
├── app.py # FastAPI + HTML/CSS frontend
├── requirements.txt # Python dependencies
├── start_app.bat # One‑click launcher
├── install_deps.bat # One‑time dependency installer
└── README.md

... (baqi content)

## Author
Mudassir Riaz

## License
MIT


- `final_model_v6` (entire folder) – 
download from [Google Drive]
(https://drive.google.com/drive/folders/13doClkiUts7PgRignssmcXbMXFTMs7c9?usp=sharing)