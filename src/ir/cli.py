"""
Command-Line Interface (CLI) for Information Retrieval & Sentiment-Aware Search.
Usage:
    python -m src.ir.cli "cyberpunk update" --k 10 --model bm25
    python -m src.ir.cli "xbox" --sentiment Positive --k 5
    python -m src.ir.cli "\"call of duty\"" --k 5
    python -m src.ir.cli "ps5 AND playstation" --query-type boolean
"""

import argparse
import sys
from pathlib import Path

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ir.search import SearchEngine


def run_cli():
    parser = argparse.ArgumentParser(
        description="Sentiment-Aware Tweet Search CLI (Information Retrieval Mini Project)"
    )
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument("--k", type=int, default=10, help="Number of results to retrieve (default: 10)")
    parser.add_argument("--model", type=str, choices=["bm25", "tfidf"], default="bm25", help="Ranking model: bm25 or tfidf")
    parser.add_argument("--sentiment", type=str, choices=["Positive", "Negative", "All"], default=None, help="Filter by predicted sentiment")
    parser.add_argument("--entity", type=str, default=None, help="Filter by entity name")
    parser.add_argument("--expansion", action="store_true", help="Enable Rocchio pseudo-relevance feedback")
    parser.add_argument("--query-type", type=str, choices=["auto", "free_text", "phrase", "boolean"], default="auto", help="Query interpretation mode")

    args = parser.parse_args()

    print(f"\n🔍 Initializing Search Engine...")
    engine = SearchEngine()

    print(f"🔎 Executing query: '{args.query}' [Model: {args.model.upper()}, k: {args.k}, Type: {args.query_type}]")
    if args.expansion:
        print("⚡ Rocchio query expansion: ENABLED")
    if args.sentiment:
        print(f"🎯 Sentiment filter: {args.sentiment}")
    if args.entity:
        print(f"🎯 Entity filter: {args.entity}")

    response = engine.search(
        query=args.query,
        k=args.k,
        model=args.model,
        sentiment_filter=args.sentiment,
        entity_filter=args.entity,
        use_query_expansion=args.expansion,
        query_type=args.query_type
    )

    results = response["results"]
    print(f"\n⏱️ Retrieved {len(results)} of {response['total_matching']} matching documents in {response['latency_ms']:.2f} ms")
    print(f"📊 Sentiment Distribution (Top-{len(results)}): {response['sentiment_distribution']}")
    if response["expanded_terms"]:
        print(f"✨ Expanded Query Terms: {response['expanded_terms']}")

    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'Score':<7} | {'Entity':<20} | {'Sentiment':<10} | {'Conf':<5} | {'Tweet Snippet'}")
    print("-" * 90)

    for item in results:
        rank = item["rank"]
        score = f"{item['score']:.4f}"
        entity = item["entity"][:20]
        sentiment = item["predicted_sentiment"]
        conf = f"{item['sentiment_confidence']:.2f}"
        snippet = item["text"].replace("\n", " ")[:40] + ("..." if len(item["text"]) > 40 else "")
        print(f"{rank:<3} | {score:<7} | {entity:<20} | {sentiment:<10} | {conf:<5} | {snippet}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_cli()
