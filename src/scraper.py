import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


HN_SEARCH_API = "https://hn.algolia.com/api/v1/search_by_date"

CATEGORY_KEYWORDS = {
    "ai": ["ai", "llm", "gpt", "ml", "machine learning", "genai", "agent"],
    "automation": ["automation", "automate", "workflow", "no-code"],
    "developer_tools": ["developer", "devtool", "sdk", "api", "cli", "ide", "code"],
    "productivity": ["productivity", "notes", "calendar", "task", "todo"],
    "saas": ["saas", "platform", "cloud", "b2b"],
}


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def infer_categories(title: str, text: str) -> list[str]:
    blob = f"{title} {text}".lower()
    categories: list[str] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in blob for keyword in keywords):
            categories.append(category)
    return categories or ["general"]


def parse_product_name(title: str | None) -> str:
    if not title:
        return "unknown_product"
    cleaned = re.sub(r"^show hn\s*:\s*", "", title.strip(), flags=re.IGNORECASE)
    return cleaned or "unknown_product"


def request_with_retry(session: requests.Session, params: dict[str, Any], retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(HN_SEARCH_API, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            last_error = error
            sleep_seconds = 2**attempt
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed request after {retries} retries: {last_error}")


def scrape_products(limit: int, rate_limit_seconds: float, retries: int) -> list[dict[str, Any]]:
    session = requests.Session()
    collected: list[dict[str, Any]] = []
    page = 0

    while len(collected) < limit:
        payload = request_with_retry(
            session,
            {
                "query": "Show HN",
                "tags": "story",
                "hitsPerPage": 100,
                "page": page,
            },
            retries=retries,
        )

        hits = payload.get("hits", [])
        if not hits:
            break

        for item in hits:
            title = item.get("title") or ""
            if "show hn" not in title.lower():
                continue

            product_name = parse_product_name(title)
            story_text = clean_html(item.get("story_text"))
            tagline = story_text[:240].strip() if story_text else product_name
            tags = infer_categories(title, story_text)

            popularity_signal = {
                "points": int(item.get("points") or 0),
                "num_comments": int(item.get("num_comments") or 0),
            }

            product_url = item.get("url") or f"https://news.ycombinator.com/item?id={item.get('objectID', '')}"

            record = {
                "product_name": product_name,
                "tagline": tagline,
                "tags": tags,
                "popularity_signal": popularity_signal,
                "product_url": product_url,
                "scrape_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source": "hacker_news_show_hn",
                "source_id": item.get("objectID", ""),
                "raw_title": title,
            }

            if not record["product_name"]:
                record["product_name"] = "unknown_product"
            if not isinstance(record["tags"], list):
                record["tags"] = ["general"]

            collected.append(record)
            if len(collected) >= limit:
                break

        page += 1
        time.sleep(rate_limit_seconds)

    return collected[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape product listings from public platform data")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=str, default="data/raw/products_raw.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    products = scrape_products(limit=args.limit, rate_limit_seconds=args.rate_limit, retries=args.retries)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(products, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(products)} products to {output_path}")


if __name__ == "__main__":
    main()
