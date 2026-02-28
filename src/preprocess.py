import argparse
import ast
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer


FALLBACK_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "with",
    "by",
    "this",
    "that",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "it",
    "from",
    "we",
    "you",
    "your",
    "our",
}


def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def preprocess_text(text: str, stemmer: PorterStemmer, stopwords: set[str]) -> tuple[str, list[str]]:
    text = normalize_unicode(text)
    text = text.lower()
    text = strip_html(text)
    text = remove_urls(text)
    text = re.sub(r"[^\w\s]", " ", text)

    raw_tokens = re.findall(r"\b\w+\b", text)
    processed_tokens: list[str] = []
    for token in raw_tokens:
        if token in stopwords:
            continue
        if token.isnumeric():
            continue
        if len(token) < 2:
            continue
        processed_tokens.append(stemmer.stem(token))

    text_clean = " ".join(processed_tokens)
    return text_clean, processed_tokens


def load_raw(input_path: Path) -> pd.DataFrame:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess product listing texts")
    parser.add_argument("--input", type=str, default="data/raw/products_raw.json")
    parser.add_argument("--output", type=str, default="data/processed/products_clean.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(input_path)
    for col in ["product_name", "tagline"]:
        if col not in df.columns:
            df[col] = ""

    stemmer = PorterStemmer()
    stopwords = FALLBACK_STOPWORDS

    text_raw_list: list[str] = []
    text_clean_list: list[str] = []
    tokens_list: list[list[str]] = []
    token_count_list: list[int] = []

    for _, row in df.iterrows():
        product_name = str(row.get("product_name") or "")
        tagline = str(row.get("tagline") or "")
        text_raw = f"{product_name} {tagline}".strip()
        text_clean, tokens = preprocess_text(text_raw, stemmer=stemmer, stopwords=stopwords)

        text_raw_list.append(text_raw)
        text_clean_list.append(text_clean)
        tokens_list.append(tokens)
        token_count_list.append(len(tokens))

    out_df = df.copy()
    out_df["text_raw"] = text_raw_list
    out_df["text_clean"] = text_clean_list
    out_df["tokens"] = [json.dumps(tokens) for tokens in tokens_list]
    out_df["token_count"] = token_count_list

    out_df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
