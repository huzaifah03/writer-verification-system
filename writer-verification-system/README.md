# Writer Verification System

An AI-based web application that detects whether two handwritten assignments were written by the same person. Built for FYP-2 at Iqra University.

---

## Project Overview

The system compares two handwritten assignment images using a **hybrid feature extraction pipeline**:
- **CNN features** — ResNet50 (pretrained on ImageNet) extracts deep visual features
- **HOG features** — Histogram of Oriented Gradients captures stroke direction and texture
- **Cosine similarity** — produces a score (0–1) and a Same Writer / Different Writer decision

**Tech Stack:** Python · Flask · PyTorch · OpenCV · SQLite · HTML/CSS/JS

---

## Repository Structure

```
writer-verification-system/
├── app/                    # Flask web application
│   ├── static/             # CSS, JS
│   ├── templates/          # HTML pages (base, index, history, result)
│   ├── routes.py           # API endpoints and page routes
│   └── __init__.py         # App factory
├── model/                  # AI pipeline
│   ├── preprocessing.py    # Grayscale, denoise, binarize, resize
│   ├── feature_extractor.py# CNN (ResNet50) + HOG hybrid features
│   ├── similarity.py       # Cosine similarity + decision logic
│   └── predict.py          # Main inference entry point
├── training/
│   └── train.py            # Training script (local GPU / Colab / Kaggle)
├── database/
│   └── db.py               # SQLite models (Assignment, VerificationResult)
├── uploads/                # Uploaded images (gitignored)
├── saved_models/           # Trained model weights (gitignored)
├── config.py               # App configuration and constants
├── run.py                  # Flask app entry point
└── requirements.txt
```

---

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, working code only |
| `dev` | Integration branch — merge features here first |
| `feature/preprocessing` | Member 1 — preprocessing & feature extraction |
| `feature/model` | Member 2 — training script & model |
| `feature/frontend` | Member 3 — Flask routes & UI |

**Workflow:** `feature/*` → PR into `dev` → reviewed → merged into `main`

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/huzaifah03/writer-verification-system.git
cd writer-verification-system
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python run.py
```
Open your browser at: **http://localhost:5000**

---

## Training the Model

### Local GPU (GTX 1660 Ti or better)
```bash
python training/train.py --data_dir data/processed --epochs 20 --batch_size 32
```

### Google Colab / Kaggle
Open `training/train.ipynb` and follow the notebook cells. The notebook auto-detects GPU.

The trained model is saved to `saved_models/writer_verification_model.pth`.

---

## Dataset

- **IAM Handwriting Dataset** — benchmark offline handwriting dataset
- **Self-collected samples** — handwriting samples collected from participants

Dataset is NOT committed to git (too large). Store locally in `data/` folder.

---

## Team

| Member | Responsibility |
|--------|---------------|
| Member 1 | Preprocessing module, feature extraction |
| Member 2 | Model training, similarity module |
| Member 3 | Flask backend, frontend UI |

**Supervisor:** Ms. Iqra Kamal — Iqra University
