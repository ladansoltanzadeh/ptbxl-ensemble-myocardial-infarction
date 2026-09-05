# A Reproducible Ensemble Learning Framework for 12‑Lead ECG Classification of Myocardial Ischemia and Infarction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22347791.svg)](https://doi.org/10.5281/zenodo.22347791)

---

## 📌 Overview

This repository contains the complete, reproducible implementation of a systematic hyperparameter search and ensemble learning framework for binary classification of **myocardial infarction (MI) and ischemia** versus other ECG findings using 12‑lead ECG signals from the **PTB‑XL** dataset.

The framework includes:

- **Five Teacher architecture variants** with different hyperparameter configurations (hidden dimension, transformer layers, dropout, learning rate).
- **Extended training variants (V2, V3)** with increased positive‑class weighting to improve sensitivity.
- **Ensemble construction** via logit averaging (V1+V2) for improved generalization.
- **Comprehensive evaluation** including accuracy, sensitivity, specificity, F1‑score, and AUC.

**Key Results:**
- **Best single architecture (V1):** 86.58% validation accuracy
- **Final Ensemble (V1+V2):** **87.40% test accuracy**, **76.18% sensitivity**, **92.65% specificity**, **93.97% AUC**

---

## 📁 Repository Structure
ptbxl-ensemble-myocardial-infarction/
├── src/ # All Python source code
│ ├── auto_train_teacher.py # Teacher training + hyperparameter search
│ ├── ensemble_teacher.py # Ensemble construction (V1+V2)
│ ├── generate_paper1_.py # Tables and figures for the main paper
│ ├── generate_paper2_.py # Tables and figures for supplementary
│ └── ...
├── results/
│ ├── paper1/ # Tables and figures for the main paper
│ ├── paper2/ # Supplementary tables and figures
│ ├── student/ # Student model evaluation results
│ └── auto_teacher/ # Teacher trial results and ensemble metrics
├── figures/ # High‑resolution figures used in the paper
├── hyperparameters.json # Complete hyperparameter search configuration
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment specification
├── run_pipeline.py # Master script to reproduce all results
├── CITATION.cff # Citation metadata
└── README.md # This file

text

---

## 📊 Data Policy

This repository intentionally contains **no PTB‑XL signals, metadata tables, checkpoints, or other large generated files**.  
Download PTB‑XL separately from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and place it under `C:\ptbxl\data\ptbxl\` (or adapt the configured root paths).

### Split and Evaluation Policy

- **Training:** official folds 1–8
- **Validation/model selection:** fold 9
- **Final test:** fold 10
- Patient‑disjoint integrity is maintained across all splits.
- The test fold remains sealed until the teacher, threshold, and ensemble are finalized.
- All reported results are based on a **single final evaluation on the test set**.

---

## 🛠️ Setup and Installation

### Windows Setup

1. **Clone the repository:**
   ```cmd
   git clone https://github.com/ladansoltanzadeh/ptbxl-ensemble-myocardial-infarction.git
   cd ptbxl-ensemble-myocardial-infarction
Create a virtual environment (recommended):

cmd
python -m venv ecg_env
ecg_env\Scripts\activate
Install dependencies:

cmd
pip install -r requirements.txt
Or using Conda:

cmd
conda env create -f environment.yml
conda activate ecg_env
Download the PTB‑XL dataset from PhysioNet and place it in C:\ptbxl\data\ptbxl\.

🚀 How to Reproduce the Results
Run the Full Pipeline (Recommended)
cmd
python run_pipeline.py
This master script executes the entire pipeline: data preparation → Teacher training → Ensemble construction → evaluation → figure generation.

Run Individual Steps (Optional)
Step	Command
Teacher Training	python src/auto_train_teacher.py --config hyperparameters.json --output results/auto_teacher --resume
Teacher Ensemble	python src/ensemble_teacher.py
Generate Paper Figures	python src/generate_paper1_figures_tables.py
📊 Key Results
Model / Ensemble	Accuracy (%)	Sensitivity (%)	Specificity (%)	AUC (%)
V1+V2 Ensemble (Final)	87.40	76.18	92.65	93.97
V1 Alone	86.08	76.46	90.58	92.94
V2 (extended training)	85.39	88.82	84.50	–
V3 (extended training)	83.01	93.53	79.44	–
For full details, please refer to the paper and the tables in results/paper1/.

📖 Citation
If you use this code or the results in your research, please cite it as:

bibtex
@software{soltanzadeh_2026_ecg_ensemble,
  author = {Soltanzadeh, Ladan and Babazadeh Sangar, Amin and Majidzadeh, Kambiz and Hosseinpour, Vahid},
  title = {A Reproducible Ensemble Learning Framework for 12‑Lead ECG Classification of Myocardial Ischemia and Infarction},
  year = {2026},
  publisher = {Zenodo},
  version = {1.0.0},
  doi = {10.5281/zenodo.22347791},
  url = {https://doi.org/10.5281/zenodo.22347791}
}
You can also cite the GitHub repository directly:

bibtex
@misc{soltanzadeh_2026_github,
  author = {Soltanzadeh, Ladan and Babazadeh Sangar, Amin and Majidzadeh, Kambiz and Hosseinpour, Vahid},
  title = {A Reproducible Ensemble Learning Framework for 12‑Lead ECG Classification of Myocardial Ischemia and Infarction},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ladansoltanzadeh/ptbxl-ensemble-myocardial-infarction}
}
A CITATION.cff file is also available in the root directory.

📬 Contact
Corresponding Author:
Amin Babazadeh Sangar
📧 aminbzh@iau.ac.ir

Affiliation:
Department of Computer Engineering, Urmia Branch, Islamic Azad University, Urmia, Iran

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🔗 Links
GitHub Repository: https://github.com/ladansoltanzadeh/ptbxl-ensemble-myocardial-infarction

Zenodo DOI: 10.5281/zenodo.22347791

PTB‑XL Dataset: PhysioNet

Last Updated: 2026-09-05
