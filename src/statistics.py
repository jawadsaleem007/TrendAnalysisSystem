import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_tokens(token_str: str) -> list[str]:
    try:
        return json.loads(token_str)
    except Exception:
        return []


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) == 0:
        return len(right)
    if len(right) == 0:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            cost = 0 if char_left == char_right else 1
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def detect_near_duplicates(titles: list[str], threshold: int = 3, max_pairs: int = 25) -> list[dict[str, int | str]]:
    pairs: list[dict[str, int | str]] = []
    normalized = [title.strip().lower() for title in titles if title and title.strip()]

    for i in range(len(normalized)):
        left = normalized[i]
        for j in range(i + 1, len(normalized)):
            right = normalized[j]
            if abs(len(left) - len(right)) > threshold:
                continue
            distance = levenshtein_distance(left, right)
            if distance <= threshold:
                pairs.append({"title_a": left, "title_b": right, "distance": distance})
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def estimate_unigram_probabilities(unigram_counts: Counter) -> dict[str, float]:
    total = sum(unigram_counts.values())
    if total == 0:
        return {}
    return {token: count / total for token, count in unigram_counts.items()}


def compute_perplexity_for_doc(tokens: list[str], probs: dict[str, float], vocab_size: int, total_count: int) -> float:
    if not tokens:
        return 0.0

    log_prob_sum = 0.0
    denominator = total_count + vocab_size
    for token in tokens:
        token_count = probs.get(token, 0.0) * total_count
        smoothed_prob = (token_count + 1.0) / denominator
        log_prob_sum += math.log(smoothed_prob)

    avg_negative_log = -log_prob_sum / len(tokens)
    return math.exp(avg_negative_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute linguistic statistics and summary report")
    parser.add_argument("--raw", type=str, default="data/raw/products_raw.json")
    parser.add_argument("--processed", type=str, default="data/processed/products_clean.csv")
    parser.add_argument("--vocab", type=str, default="data/features/vocab.json")
    parser.add_argument("--freq", type=str, default="data/features/frequencies.json")
    parser.add_argument("--output", type=str, default="reports/trend_summary.txt")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    processed_path = Path(args.processed)
    vocab_path = Path(args.vocab)
    freq_path = Path(args.freq)
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)
    raw_df = pd.DataFrame(raw_data)

    processed_df = pd.read_csv(processed_path)
    token_docs = [parse_tokens(str(item)) for item in processed_df["tokens"].fillna("[]").tolist()]

    with vocab_path.open("r", encoding="utf-8") as file:
        vocab_data = json.load(file)
    with freq_path.open("r", encoding="utf-8") as file:
        freq_data = json.load(file)

    unigram_top_30 = freq_data.get("unigrams", [])[:30]
    bigram_top_20 = freq_data.get("bigrams", [])[:20]

    tags_counter = Counter()
    for tags in raw_df.get("tags", []):
        if isinstance(tags, list):
            tags_counter.update(tags)
        elif isinstance(tags, str):
            try:
                parsed = json.loads(tags)
                if isinstance(parsed, list):
                    tags_counter.update(parsed)
            except Exception:
                continue

    vocabulary_size = int(vocab_data.get("vocab_size", 0))
    avg_desc_length = float(processed_df["token_count"].fillna(0).mean()) if len(processed_df) else 0.0

    title_series = raw_df.get("product_name", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    duplicate_pairs = detect_near_duplicates(title_series, threshold=3)

    unigram_counts = Counter({item["token"]: int(item["count"]) for item in freq_data.get("unigrams", [])})
    unigram_probs = estimate_unigram_probabilities(unigram_counts)
    total_count = sum(unigram_counts.values())

    held_out_docs = token_docs[:5]
    perplexities = [
        compute_perplexity_for_doc(doc, unigram_probs, vocab_size=max(1, vocabulary_size), total_count=max(1, total_count))
        for doc in held_out_docs
    ]

    lines: list[str] = []
    lines.append("TrendScope NLP Trend Summary")
    lines.append("=" * 40)
    lines.append("")

    lines.append("1) Top 30 Unigrams")
    for item in unigram_top_30:
        lines.append(f"- {item['token']}: {item['count']}")
    lines.append("")

    lines.append("2) Top 20 Bigrams")
    for item in bigram_top_20:
        lines.append(f"- {item['token']}: {item['count']}")
    lines.append("")

    lines.append("3) Most Common Tags/Categories")
    for tag, count in tags_counter.most_common(20):
        lines.append(f"- {tag}: {count}")
    lines.append("")

    lines.append(f"4) Vocabulary Size: {vocabulary_size}")
    lines.append(f"5) Average Description Length (tokens): {avg_desc_length:.2f}")
    lines.append("")

    lines.append("6) Near-Duplicate Titles (MED threshold <= 3)")
    if duplicate_pairs:
        for pair in duplicate_pairs:
            lines.append(f"- ({pair['distance']}) {pair['title_a']} <-> {pair['title_b']}")
    else:
        lines.append("- No near-duplicates found within threshold.")
    lines.append("")

    lines.append("7) Unigram Probability Estimates (Top 30 by probability)")
    top_probabilities = sorted(unigram_probs.items(), key=lambda item: item[1], reverse=True)[:30]
    for token, prob in top_probabilities:
        lines.append(f"- P({token}) = {prob:.6f}")
    lines.append("")

    lines.append("8) Perplexity for 5 Held-Out Product Descriptions")
    for index, value in enumerate(perplexities, start=1):
        lines.append(f"- Doc {index}: {value:.4f}")

    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
