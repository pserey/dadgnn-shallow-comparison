# DADGNN Shallow Comparison

A scientific replication project that provides shallow learning baselines to compare against the text classification results reported in the EMNLP 2021 paper **"DADGNN: Deep Attention Diffusion Graph Neural Networks for Text Classification"**.

This project implements classical machine learning approaches (TF-IDF + shallow models) to establish performance benchmarks on the same text classification datasets used in the original paper.

## 📊 Datasets & Models

### Datasets (3)
- **ag_news** — 4-class news topic classification 
- **trec** — 6-class question type classification (using coarse labels)
- **rotten_tomatoes** — 2-class movie review sentiment analysis

### Models (Shallow Learning Only)
- **TF-IDF + Logistic Regression** (one-vs-rest with L2 regularization)
- **TF-IDF + Linear SVM** (LinearSVC with squared hinge loss)
- **TF-IDF + Multinomial Naive Bayes**

All models use the same TF-IDF vectorization configuration:
- Lowercase normalization, unicode accent stripping
- Word-level analysis with 1-2 grams
- Min/max document frequency filtering (min_df=2, max_df=0.9)
- Maximum 100,000 features with sublinear TF scaling

## 🚀 Getting Started

### 1. Install Python with pyenv

```bash
# Install Python 3.13.9 (or your preferred 3.11+ version)
pyenv install -s 3.13.9
pyenv local 3.13.9
```

### 2. Initialize Project with uv

```bash
# Create uv project (if not already done)
uv init --package

# Add all required dependencies (latest versions)
uv add datasets scikit-learn numpy scipy pandas tqdm

# Optional: Add matplotlib for plots
uv add matplotlib
```

### 3. Project Structure

The project uses a `src/` package for all code with absolute imports:

```
dadgnn-shallow-comparison/
├── README.md
├── .python-version          # pyenv version (e.g., 3.13.9)
├── pyproject.toml           # uv project configuration (auto-managed)
├── uv.lock                  # dependency lockfile
├── src/                     # Main Python package
│   ├── __init__.py
│   ├── data.py              # Dataset loaders (Hugging Face)
│   ├── vectorizer.py        # TF-IDF configuration
│   ├── models.py            # Model pipelines & hyperparameter grids
│   ├── eval.py              # Metrics: accuracy, F1, bootstrap CI, confusion matrix
│   └── run_experiment.py    # CLI entrypoint
└── results/
    ├── tables/
    └── logs/
```

## 🎯 Usage

### Run Single Experiment

```bash
# Basic usage
uv run python -m src.run_experiment --dataset ag_news --model logistic_regression

# All dataset options
uv run python -m src.run_experiment --dataset ag_news --model linear_svm
uv run python -m src.run_experiment --dataset trec --model multinomial_nb
uv run python -m src.run_experiment --dataset rotten_tomatoes --model logistic_regression
```

### Available Options

**Datasets:**
- `ag_news` (4 classes)
- `trec` (6 classes) 
- `rotten_tomatoes` (2 classes)

**Models:**
- `logistic_regression` (TF-IDF + Logistic Regression)
- `linear_svm` (TF-IDF + Linear SVM)
- `multinomial_nb` (TF-IDF + Multinomial Naive Bayes)

### Complete Experimental Protocol

Each experiment automatically performs:

1. **Dataset Loading**: Downloads via Hugging Face `datasets` library
2. **Data Splitting**: Uses official splits; creates 10% validation split if needed
3. **Hyperparameter Search**: GridSearchCV with 5-fold stratified cross-validation
   - Selection criterion: Macro-F1 score
   - Searches over predefined parameter grids for each model
4. **Final Training**: Retrains best model on combined train+validation data
5. **Test Evaluation**: Reports accuracy, 95% bootstrap confidence interval, macro-F1, confusion matrix

### Example Output

```
============================================================
EVALUATION RESULTS
============================================================
Dataset: ag_news
Model: TF-IDF + Logistic Regression
Test samples: 7600
Classes: 4

METRICS:
  Accuracy: 0.9237
  95% CI: (0.9171, 0.9301)
  Macro-F1: 0.9235

Confusion Matrix:
True\Pred   Class 0  Class 1  Class 2  Class 3
Class 0        1847        43        70        40
Class 1          38      1862        87        13  
Class 2          51        81      1804        64
Class 3          44        19        59      1878
============================================================

BEST HYPERPARAMETERS:
  classifier__C: 2
  classifier__class_weight: balanced
  classifier__penalty: l2
  classifier__max_iter: 2000
```

## 🔧 Hyperparameter Grids

### Logistic Regression
- `C`: [0.5, 1, 2, 4]
- `class_weight`: [None, "balanced"]  
- `penalty`: "l2"
- `max_iter`: 2000

### Linear SVM
- `C`: [0.5, 1, 2, 4]
- `class_weight`: [None, "balanced"]
- `loss`: "squared_hinge"

### Multinomial Naive Bayes  
- `alpha`: [0.1, 0.5, 1.0, 2.0]

## 📈 Evaluation Metrics

- **Primary**: Accuracy with 95% bootstrap confidence interval (1000 resamples)
- **Secondary**: Macro-averaged F1 score
- **Diagnostic**: Confusion matrix
- **Statistical**: Bootstrap confidence intervals for robust performance estimation

## 🏗️ Architecture Notes

- **Package Management**: Uses `uv` exclusively (no manual `pyproject.toml` editing)
- **Import Strategy**: All imports use absolute paths (`from src.module import ...`)  
- **CLI Interface**: Entry point via `python -m src.run_experiment`
- **Reproducibility**: Fixed random seeds (42) throughout pipeline
- **Performance**: Utilizes `n_jobs=-1` for parallel processing where applicable

## 📚 References

Chen, Y., Chen, D., Liu, Z., Zhang, Z., & Wang, C. (2021). *DADGNN: Deep Attention Diffusion Graph Neural Networks for Text Classification*. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).

---

**Note**: This is a shallow learning baseline project. No deep learning models (PyTorch, TensorFlow, etc.) or neural architectures are implemented. The goal is to establish classical ML performance benchmarks for comparison with the original DADGNN results.
