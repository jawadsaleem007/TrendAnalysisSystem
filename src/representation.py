import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_tokens(token_str: str) -> list[str]:
    try:
        return json.loads(token_str)
    except Exception:
        return []


def build_bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def one_hot_encode(doc_tokens: list[list[str]], vocab_index: dict[str, int]) -> list[list[int]]:
    one_hot_rows: list[list[int]] = []
    vocab_size = len(vocab_index)
    for tokens in doc_tokens:
        row = [0] * vocab_size
        for token in set(tokens):
            if token in vocab_index:
                row[vocab_index[token]] = 1
        one_hot_rows.append(row)
    return one_hot_rows


def bag_of_words(doc_tokens: list[list[str]], vocab_index: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(doc_tokens), len(vocab_index)), dtype=np.int32)
    for i, tokens in enumerate(doc_tokens):
        counts = Counter(tokens)
        for token, count in counts.items():
            token_id = vocab_index.get(token)
            if token_id is not None:
                matrix[i, token_id] = count
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NLP representations")
    parser.add_argument("--input", type=str, default="data/processed/products_clean.csv")
    parser.add_argument("--vocab-out", type=str, default="data/features/vocab.json")
    parser.add_argument("--bow-out", type=str, default="data/features/bow_matrix.npy")
    parser.add_argument("--freq-out", type=str, default="data/features/frequencies.json")
    parser.add_argument("--one-hot-size", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    vocab_path = Path(args.vocab_out)
    bow_path = Path(args.bow_out)
    freq_path = Path(args.freq_out)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    doc_tokens = [parse_tokens(str(item)) for item in df["tokens"].fillna("[]").tolist()]

    unigram_counter = Counter()
    bigram_counter = Counter()
    for tokens in doc_tokens:
        unigram_counter.update(tokens)
        bigram_counter.update(build_bigrams(tokens))

    vocab = sorted(unigram_counter.keys())
    vocab_index = {token: idx for idx, token in enumerate(vocab)}

    bow = bag_of_words(doc_tokens=doc_tokens, vocab_index=vocab_index)
    np.save(bow_path, bow)

    subset = doc_tokens[: max(1, min(args.one_hot_size, len(doc_tokens)))]
    one_hot_subset = one_hot_encode(subset, vocab_index)

    vocab_payload = {
        "vocab_size": len(vocab),
        "vocab": vocab,
        "token_to_index": vocab_index,
        "one_hot_subset_doc_count": len(one_hot_subset),
        "one_hot_subset": one_hot_subset,
    }

    with vocab_path.open("w", encoding="utf-8") as file:
        json.dump(vocab_payload, file, ensure_ascii=False, indent=2)

    freq_payload = {
        "unigrams": [{"token": token, "count": count} for token, count in unigram_counter.most_common()],
        "bigrams": [
            {"token": f"{left} {right}", "count": count}
            for (left, right), count in bigram_counter.most_common()
        ],
    }
    with freq_path.open("w", encoding="utf-8") as file:
        json.dump(freq_payload, file, ensure_ascii=False, indent=2)

    print(
        f"Saved vocab ({len(vocab)} terms) to {vocab_path}, BoW matrix {bow.shape} to {bow_path}, frequencies to {freq_path}"
    )


if __name__ == "__main__":
    main()
