"""
Command-line interface for tweet sentiment analysis.

Examples
--------
Analyze a single tweet:
    python -m src.sentiment_cli "I love the new Battlefield update!"

Pick a specific model:
    python -m src.sentiment_cli "The servers are down again..." --model logistic_regression

Batch-analyze a CSV file:
    python -m src.sentiment_cli --file data/tweets.csv --output results/predictions.csv

Run an interactive prompt when no arguments are supplied:
    python -m src.sentiment_cli
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow both `python src/sentiment_cli.py` and `python -m src.sentiment_cli`.
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from config import RESULTS_DIR
    from predict import SentimentPredictor
except ImportError:  # pragma: no cover - package-style fallback
    from src.config import RESULTS_DIR
    from src.predict import SentimentPredictor


MODEL_CHOICES = {
    "ensemble": "Ensemble (Soft Voting)",
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
}

TEXT_COLUMN_CANDIDATES = ("text", "tweet", "tweet_text", "content", "message")
LABEL_COLUMN_CANDIDATES = ("sentiment", "label", "target")


def _format_result(result: dict) -> str:
    return f"{result['sentiment']} ({result['confidence'] * 100:.1f}% confidence)"


def _analyze_single(predictor: SentimentPredictor, text: str, as_json: bool) -> None:
    result = predictor.predict(text)
    if as_json:
        print(json.dumps({"text": text, **result}, ensure_ascii=False, indent=2))
    else:
        print(f"\nTweet: {text}")
        print(f"Predicted sentiment: {_format_result(result)}\n")


def _analyze_interactive(predictor: SentimentPredictor) -> None:
    print("Interactive sentiment analysis. Type a tweet and press Enter.")
    print("Type 'quit', 'exit', or press Ctrl-D to stop.\n")
    while True:
        try:
            text = input("Tweet> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        result = predictor.predict(text)
        print(f"  -> {_format_result(result)}\n")


def _resolve_column(columns, candidates, explicit=None) -> str | None:
    if explicit:
        if explicit not in columns:
            raise ValueError(
                f"Column '{explicit}' not found. Available columns: {list(columns)}"
            )
        return explicit
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _analyze_file(
    predictor: SentimentPredictor,
    file_path: Path,
    output_path: Path,
    text_column: str | None,
    label_column: str | None,
) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    text_col = _resolve_column(df.columns, TEXT_COLUMN_CANDIDATES, text_column)

    # The bundled `data/tweets.csv` has no header row (id, entity, sentiment, text).
    # Fall back to the same headerless layout used by src/preprocess.load_corpus().
    if text_col is None and text_column is None:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] >= 4:
            df.columns = ["id", "entity", "sentiment", "text"][: df.shape[1]]
        elif df.shape[1] == 2:
            df.columns = ["text", "sentiment"]
        text_col = _resolve_column(df.columns, TEXT_COLUMN_CANDIDATES)

    if text_col is None:
        raise ValueError(
            "Could not detect a text column. Available columns: "
            f"{list(df.columns)}. Use --text-column to specify one."
        )

    print(f"Analyzing {len(df):,} rows from {file_path} using column '{text_col}'...")
    results = [predictor.predict(str(text)) for text in df[text_col].fillna("")]
    df["predicted_sentiment"] = [r["sentiment"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    counts = df["predicted_sentiment"].value_counts()
    print("\nPredicted sentiment distribution:")
    for label, count in counts.items():
        print(f"  {label:<10} {count:>7} ({count / len(df) * 100:5.1f}%)")

    # If the file already contains ground-truth labels, report accuracy.
    # The classifier is binary, so only Positive/Negative rows are scored;
    # Neutral/Irrelevant labels are reported separately rather than counted as errors.
    label_col = _resolve_column(df.columns, LABEL_COLUMN_CANDIDATES, label_column)
    if label_col and label_col != "predicted_sentiment":
        labels = df[label_col].astype("string").str.strip().str.title()
        labelled = labels.notna()
        binary_mask = labels.isin(["Positive", "Negative"])
        if binary_mask.any():
            accuracy = (labels[binary_mask] == df.loc[binary_mask, "predicted_sentiment"]).mean()
            print(
                f"\nAccuracy on {int(binary_mask.sum()):,} Positive/Negative rows "
                f"('{label_col}'): {accuracy * 100:.2f}%"
            )
        other_labels = int((labelled & ~binary_mask).sum())
        if other_labels:
            print(f"Skipped {other_labels:,} rows with Neutral/Irrelevant labels (out of binary scope).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tweet sentiment analysis (Positive vs Negative) from the command line."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Tweet text to analyze. Omit to start an interactive prompt.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="CSV file with tweets to classify in batch.",
    )
    parser.add_argument(
        "--text-column",
        help="Name of the CSV column containing tweet text (auto-detected by default).",
    )
    parser.add_argument(
        "--label-column",
        help="Optional ground-truth label column used to report batch accuracy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "sentiment_predictions.csv",
        help="Output CSV path for batch predictions (default: results/sentiment_predictions.csv).",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CHOICES),
        default="ensemble",
        help="Classifier to use (default: ensemble).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print single-tweet results as JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.file and args.text:
        print("Error: provide either tweet text or --file, not both.", file=sys.stderr)
        return 2

    try:
        predictor = SentimentPredictor(model_type=args.model)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Train the models first: python src/train.py", file=sys.stderr)
        return 1

    try:
        if args.file:
            _analyze_file(
                predictor,
                args.file,
                args.output,
                args.text_column,
                args.label_column,
            )
        elif args.text:
            _analyze_single(predictor, " ".join(args.text).strip(), args.json)
        else:
            _analyze_interactive(predictor)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
