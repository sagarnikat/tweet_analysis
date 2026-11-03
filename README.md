# 🐦 Tweet Sentiment Analysis & Information Retrieval System

An end-to-end university project that **classifies tweet sentiment** and **searches 73,000+ tweets** with a custom
Information Retrieval (IR) engine. It combines classical NLP, machine learning, hand-built ranking algorithms, a
command-line interface, and an interactive Streamlit dashboard.

The project covers two complementary tasks:

1. **Sentiment analysis** — predict whether a tweet is **Positive** or **Negative**, with confidence scores and a
   feedback loop for collecting corrections.
2. **Sentiment-aware search** — index the tweet collection, rank results with **BM25** or **TF-IDF cosine similarity**,
   and filter retrieved tweets by predicted sentiment.

---

## ✨ Features

| Area | What is included |
| :--- | :--- |
| **Text preprocessing** | URL/mention/hashtag handling, punctuation normalization, stop-word removal, negation preservation (`not`, `never`, `no`, ...), WordNet lemmatization |
| **Sentiment features** | 10,000 TF-IDF features (uni-, bi-, and tri-grams) + 6 VADER/punctuation features |
| **Classifiers** | Logistic Regression, Multinomial Naive Bayes, and a soft-voting ensemble |
| **Search index** | Custom inverted index with term frequencies, document frequencies, document lengths, and positional postings |
| **Ranking** | Okapi BM25, TF-IDF cosine similarity (scikit-learn parity), and Rocchio pseudo-relevance feedback |
| **Query modes** | Free text, exact phrase (`"call of duty"`), and Boolean (`AND`, `OR`, `NOT`, parentheses) |
| **Sentiment-aware retrieval** | Every retrieved tweet is classified in real time; results can be filtered to Positive or Negative |
| **Evaluation** | 29 unit tests, classifier reports/confusion matrices, P@k, Recall@k, MAP, MRR, nDCG@10, and 11-point PR curves |
| **Interfaces** | Streamlit web app, sentiment CLI (single/batch/interactive), and search CLI |

---

## 📊 Dataset

The bundled dataset is `data/tweets.csv`. It has **no header row** and four columns:

```text
id, entity, sentiment, text
```

| Property | Value |
| :--- | :--- |
| Total rows | **74,682** |
| Entities | **32** (games, consoles, brands) |
| Non-empty tweets indexed | **73,824** |
| Class distribution | Negative 22,542 · Positive 20,832 · Neutral 18,318 · Irrelevant 12,990 |

> **Scope note:** the sentiment classifier is **binary** (Positive vs Negative), which is the standard setup for this
> dataset. Neutral and Irrelevant tweets remain in the search corpus but are not used to train the classifier, and
> they are not counted as errors during evaluation. The IR benchmark uses `data/ir_queries.csv` (48 curated queries).

---

## 🏗️ Architecture

```text
                         ┌─────────────────────────────────────────────┐
                         │              data/tweets.csv                │
                         │      id · entity · sentiment · text         │
                         └───────────────────┬─────────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    │                                                 │
                    ▼                                                 ▼
        ┌───────────────────────┐                       ┌──────────────────────────┐
        │  Sentiment pipeline   │                       │  Information Retrieval   │
        │  src/train.py         │                       │  src/ir/index.py         │
        │                       │                       │                          │
        │  clean_text()         │                       │  clean_text()/tokenize() │
        │  TF-IDF + VADER       │                       │  inverted index          │
        │  LR / NB / Ensemble   │                       │  positional postings     │
        └───────────┬───────────┘                       └────────────┬─────────────┘
                    │                                                │
                    ▼                                                ▼
        ┌───────────────────────┐                       ┌──────────────────────────┐
        │ models/*.joblib       │                       │ models/ir_index.joblib   │
        │ predicted label +     │◄──────────────────────┤ BM25 / TF-IDF / Rocchio  │
        │ confidence            │  classify top-k docs  │ Boolean / phrase queries │
        └───────────┬───────────┘                       └────────────┬─────────────┘
                    │                                                │
                    └────────────────────┬───────────────────────────┘
                                         ▼
                     ┌─────────────────────────────────────────┐
                     │ Streamlit app · sentiment CLI · search CLI │
                     └─────────────────────────────────────────┘
```

### Key modules

| File | Responsibility |
| :--- | :--- |
| `src/config.py` | Central path registry and hyperparameters |
| `src/preprocess.py` | Cleaning, tokenization, lemmatization, VADER feature extraction |
| `src/train.py` | Deduplication, stratified split, model training, artifact saving |
| `src/evaluate.py` | Classification reports, confusion matrices, accuracy comparison |
| `src/predict.py` | `SentimentPredictor` inference engine |
| `src/sentiment_cli.py` | Command-line sentiment analysis (single, batch, interactive) |
| `src/ir/index.py` | Inverted index construction and persistence |
| `src/ir/query.py` | Query parsing for free text, phrase, and Boolean queries |
| `src/ir/ranking.py` | BM25, TF-IDF cosine, and Rocchio expansion |
| `src/ir/search.py` | End-to-end sentiment-aware search engine |
| `src/ir/evaluate_ir.py` | IR benchmark and metric plots |
| `src/ir/cli.py` | Command-line search interface |
| `app/app.py` | Streamlit dashboard |
| `tests/` | 29 automated tests |

---

## 📁 Project structure

```text
tweet_analysis/
├── app/
│   └── app.py                     # Streamlit dashboard (search, sentiment, evaluation)
├── data/
│   ├── tweets.csv                 # 74,682 labelled tweets (headerless CSV)
│   ├── ir_queries.csv             # 48 proxy evaluation queries
│   ├── generate_tweets.py         # Optional mock-data generator
│   └── prepare_data.py            # Data preparation helper
├── models/                        # Generated, git-ignored artifacts
│   ├── ir_index.joblib
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── ensemble.joblib
│   ├── tfidf_vectorizer.joblib
│   └── test_data_split.joblib
├── results/                       # Generated, git-ignored reports and plots
│   ├── ir_metrics.csv
│   ├── sentiment_metrics.csv
│   ├── ir_pr_curve.png
│   ├── ir_metrics_comparison.png
│   ├── model_comparison.png
│   └── cm_*.png
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── sentiment_cli.py
│   └── ir/
│       ├── __init__.py
│       ├── index.py
│       ├── query.py
│       ├── ranking.py
│       ├── search.py
│       ├── evaluate_ir.py
│       └── cli.py
├── tests/
├── requirements.txt
├── CHANGES.md
└── README.md
```

> The `models/` and `results/` folders are generated by the scripts and are intentionally ignored by Git. Run the
> setup commands below to recreate them.

---

## 🚀 Installation

### Prerequisites

- **Python 3.10+** (tested on Python 3.14)
- `pip` and `venv`
- ~2 GB of free disk space (dataset + dependencies)
- No GPU required

### 1. Get the project and create a virtual environment

```bash
git clone <your-repository-url>
cd tweet_analysis          # or the folder name you cloned into

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. NLTK data (usually automatic)

The preprocessing module downloads `stopwords` and `wordnet` on first use. If you are offline or if automatic download
fails, run:

```bash
python -m nltk.downloader stopwords wordnet
```

---

## ⚡ Quick start

From the project root, with the virtual environment activated:

```bash
# 1. Train the sentiment models (2–5 minutes on a laptop)
python src/train.py

# 2. Build the inverted index (~10–15 seconds)
python -m src.ir.index

# 3. Launch the web dashboard
streamlit run app/app.py
```

Open <http://localhost:8501> in your browser.

If you also want the evaluation reports and plots:

```bash
python src/evaluate.py            # sentiment metrics + confusion matrices
python -m src.ir.evaluate_ir      # IR benchmark + PR curves
```

> **Order matters only for the app:** it needs both the trained models and the index. The commands above create them.

---

## 🖥️ Usage

### 1. Streamlit web app

```bash
streamlit run app/app.py
```

The app has three tabs:

1. **🔍 Search Tweets (IR Engine)** — choose BM25 or TF-IDF, query mode (auto/free text/phrase/Boolean), result count,
   sentiment filter, entity filter, and optional Rocchio expansion. Results show rank, score, entity, predicted
   sentiment with confidence, and highlighted query terms.
2. **🧠 Sentiment Analysis** — classify a custom tweet with the ensemble, Logistic Regression, or Naive Bayes model.
   If a prediction is wrong, submit the correct label; it is saved to `data/user_feedback.csv` and incorporated the
   next time `python src/train.py` runs.
3. **📊 IR & System Evaluation** — inspect the IR metric table, PR curves, sentiment metrics, accuracy chart, and
   confusion matrices.

If the app reports missing artifacts, run steps 1 and 2 from the quick start and reload.

### 2. Sentiment CLI

Analyze a single tweet:

```bash
python -m src.sentiment_cli "I absolutely love this update, best game ever!"
python -m src.sentiment_cli "The servers are down again and I hate it." --model logistic_regression
```

Get machine-readable output:

```bash
python -m src.sentiment_cli "Not bad at all, actually pretty fun" --json
```

Batch-classify a CSV (the bundled `data/tweets.csv` works without a header):

```bash
python -m src.sentiment_cli \
  --file data/tweets.csv \
  --model logistic_regression \
  --output results/sentiment_predictions.csv
```

The batch mode auto-detects a `text` column (or the bundled 4-column layout), writes
`predicted_sentiment`/`confidence` columns, prints the prediction distribution, and reports accuracy when a
Positive/Negative ground-truth column is present.

Interactive mode (no arguments):

```bash
python -m src.sentiment_cli
```

### 3. Search CLI

```bash
# Free-text BM25 search
python -m src.ir.cli "cyberpunk update" --k 10 --model bm25

# TF-IDF cosine search
python -m src.ir.cli "battlefield server" --k 5 --model tfidf

# Boolean query
python -m src.ir.cli "ps5 OR playstation" --query-type boolean

# Exact phrase query (quotes must be escaped in the shell)
python -m src.ir.cli "\"call of duty\"" --k 5

# Sentiment-filtered and entity-filtered search
python -m src.ir.cli "playstation 5" --sentiment Positive --k 5
python -m src.ir.cli "update" --entity Cyberpunk2077 --k 5

# Rocchio pseudo-relevance feedback
python -m src.ir.cli "cyberpunk update" --expansion --k 5
```

### 4. Train and evaluate

```bash
# Train all three sentiment models and refresh evaluation plots
python src/train.py

# Evaluate saved models without retraining
python src/evaluate.py

# Rebuild the inverted index
python -m src.ir.index

# Run the IR benchmark
python -m src.ir.evaluate_ir

# Run all 29 unit tests
pytest -v
```

### 5. What each script produces

| Command | Output |
| :--- | :--- |
| `python src/train.py` | `models/logistic_regression.joblib`, `naive_bayes.joblib`, `ensemble.joblib`, `tfidf_vectorizer.joblib`, `test_data_split.joblib` |
| `python src/evaluate.py` | `results/sentiment_metrics.csv`, `results/model_comparison.png`, `results/cm_*.png` |
| `python -m src.ir.index` | `models/ir_index.joblib` |
| `python -m src.ir.evaluate_ir` | `results/ir_metrics.csv`, `results/ir_pr_curve.png`, `results/ir_metrics_comparison.png` |
| `pytest -v` | Test results in the terminal |

---

## 📈 Evaluation results

These numbers were produced by the commands above on the bundled dataset. Re-running the scripts may produce small
variations depending on library versions.

### Sentiment classification

Training uses **35,727 unique cleaned tweets** after removing 6,351 duplicates. The stratified 80/20 split gives
**28,581 training** and **7,146 test** samples.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **88.62%** | 0.8855 | 0.8862 | 0.8858 | 0.8863 |
| Naive Bayes | 86.57% | 0.8656 | 0.8641 | 0.8647 | 0.8655 |
| Soft-Voting Ensemble | 87.97% | 0.8792 | 0.8788 | 0.8790 | 0.8796 |

Artifacts: `results/sentiment_metrics.csv`, `results/model_comparison.png`,
`results/cm_logistic_regression.png`, `results/cm_naive_bayes.png`, `results/cm_ensemble.png`.

### Information Retrieval benchmark

Evaluated over **48 proxy queries** across 32 entities. **Relevance is proxy ground truth:** a retrieved tweet is
considered relevant if its `entity` field matches the query's target entity. Latency is **ranking-only** (index loaded,
no sentiment inference); an end-to-end sentiment-aware search adds classification time per returned tweet.

| Ranking model | P@5 | P@10 | P@20 | Recall@10 | MAP | MRR | nDCG@10 | Ranking latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF (cosine) | 0.7792 | 0.7896 | 0.8000 | 0.0034 | 0.7025 | 0.8594 | 0.7943 | 2.29 ms |
| **BM25** | **0.8708** | **0.8771** | **0.8708** | **0.0038** | **0.7787** | **0.9010** | **0.8777** | **2.04 ms** |
| BM25 + Rocchio | 0.8708 | 0.8458 | 0.8135 | 0.0037 | 0.6409 | 0.8815 | 0.8516 | 6.20 ms |

**Interpretation**

- BM25 gives the best overall ranking quality (MAP 0.7787, MRR 0.9010, nDCG@10 0.8777).
- Rocchio expansion keeps strong top-5 precision but adds broader terms that reduce deep-rank precision.
- Recall@10 is necessarily very small: each entity has thousands of relevant tweets, so ten results cover only a
  fraction of the relevant set.

Artifacts: `results/ir_metrics.csv`, `results/ir_pr_curve.png`, `results/ir_metrics_comparison.png`.

### Tests

```text
29 passed
```

The suite covers preprocessing/negations, index postings and serialization, Boolean and phrase queries, BM25 ordering,
scikit-learn TF-IDF parity, hand-computed IR metrics, and end-to-end search.

---

## 🔬 How it works

### Preprocessing

`clean_text()` in `src/preprocess.py`:

1. lowercases text;
2. strips URLs (including `pic.twitter.com`/`t.co`) and `@mentions`;
3. converts `#hashtags` to their words instead of dropping them;
4. replaces punctuation with spaces while preserving apostrophes;
5. removes English stop words **except** negation/contrast words;
6. applies WordNet lemmatization.

The same function is used for sentiment features, indexing, and query terms, so training and retrieval always share
one vocabulary. Punctuation is normalized *before* stop-word removal, which prevents duplicates such as `game` vs
`game.` and `game!`.

### Sentiment model

- **Features**: 10,000 TF-IDF features with 1–3 grams, plus six numerical features: VADER compound/negative/neutral/
  positive scores and exclamation/question-mark counts.
- **Models**: Logistic Regression (`class_weight='balanced'`), Multinomial Naive Bayes (with negative VADER values
  clamped to zero), and a soft-voting ensemble of the two.
- **Leakage control**: duplicate cleaned texts are removed **before** the stratified train/test split.
- **Self-learning**: corrections submitted in the app are stored in `data/user_feedback.csv` and merged into the next
  training run.

### Retrieval engine

- **Inverted index**: `term → [(doc_id, tf, [positions])]`, plus document frequencies, lengths, and average length.
- **BM25**: $k_1 = 1.5$, $b = 0.75$, Robertson–Spärck Jones IDF.
- **TF-IDF cosine**: sublinear TF and smooth IDF, verified against scikit-learn in the test suite.
- **Boolean queries**: Shunting-Yard parsing with `NOT > AND > OR` precedence and parentheses.
- **Phrase queries**: consecutive positional postings.
- **Rocchio**: pseudo-relevance feedback from the top 5 documents.
- **Sentiment-aware search**: top-ranked documents are classified and can be filtered by predicted sentiment or
  entity.

---

## 🧰 Troubleshooting

| Problem | Fix |
| :--- | :--- |
| `FileNotFoundError: Models not found` | Run `python src/train.py` |
| `FileNotFoundError: Index file not found` | Run `python -m src.ir.index` |
| `LookupError: Resource wordnet not found` | `python -m nltk.downloader stopwords wordnet` |
| `streamlit: command not found` | Activate the virtual environment, then `pip install -r requirements.txt` |
| `ModuleNotFoundError` when running a script | Run commands **from the project root** and keep the virtual environment active |
| First app search is slow | The first prediction loads NLTK/VADER resources; subsequent queries are fast (warm ranking ~2 ms, warm end-to-end search ~30 ms for a few results) |
| `results/` is empty in a fresh clone | Run `python src/evaluate.py` and `python -m src.ir.evaluate_ir` |
| NLTK download fails offline | Download the corpora on a connected machine and copy `~/nltk_data` to the same location |
| Out-of-memory during training | Close other memory-heavy apps; the pipeline is CPU-only but the TF-IDF matrix is sparse and generally small |

Model and result artifacts are deliberately git-ignored. This keeps the repository lightweight; the scripts recreate
everything deterministically (with `random_state=42`).

---

## ⚠️ Limitations and future work

- The classifier is **binary** by design. Neutral and Irrelevant tweets in the dataset are excluded from training; a
  three-class extension would require a separate model and evaluation.
- Sarcasm and implicit sentiment remain difficult; VADER + n-grams help but are not a substitute for a transformer
  model.
- IR relevance is a **proxy** (entity match), not human relevance judgement. The benchmark is useful for comparing
  ranking algorithms but should not be read as a production search-quality estimate.
- The dataset and stop-word/VADER resources are English-only.
- The self-learning loop is manual: feedback is applied on the next `python src/train.py` run, not online.

Possible extensions: transformer fine-tuning (BERT/RoBERTa), neutral-class classification, per-entity sentiment
dashboards, FastAPI endpoints, Docker packaging, and human relevance judgements for IR evaluation.

---

## 🤝 Acknowledgements

Built as a university mini project using Python, scikit-learn, NLTK, VADER, Matplotlib, Seaborn, and Streamlit. The
bundled tweet dataset is used for educational purposes; check the original dataset's license before redistributing it.
