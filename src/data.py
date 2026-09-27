"""Datensammlung aus dem Amazon-Reviews-2023-Datensatz (McAuley Lab, 2023).

Die Rezensionsdateien werden direkt vom Server des McAuley Lab (UC San Diego)
gestreamt und zeilenweise gelesen. Dadurch muss nicht die gesamte (mehrere
Gigabyte grosse) Datei heruntergeladen werden: Sobald genug gueltige
Rezensionen einer Kategorie gesammelt sind, wird der Download abgebrochen.
"""
import gzip
import json

import pandas as pd
import requests

from config import CATEGORIES, SAMPLES_PER_CATEGORY

# Basis-URL der rohen Rezensionsdateien (eine .jsonl.gz-Datei je Kategorie).
BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/{category}.jsonl.gz"
)


def _collect_category(category: str, n: int) -> pd.DataFrame:
    """Laedt bis zu n gueltige Rezensionen einer Kategorie per Streaming."""
    url = BASE_URL.format(category=category)
    rows = []
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = False  # rohe .gz-Bytes beibehalten
        with gzip.GzipFile(fileobj=response.raw) as gz:
            for raw_line in gz:
                try:
                    review = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                rating = review.get("rating")
                title = (review.get("title") or "").strip()
                text = (review.get("text") or "").strip()
                # Nur Rezensionen mit gueltiger Bewertung und Text.
                if rating is None or (not title and not text):
                    continue
                rating = int(round(float(rating)))
                if rating < 1 or rating > 5:
                    continue
                rows.append(
                    {
                        "category": category,
                        "rating": rating,
                        "text": (title + " " + text).strip(),
                    }
                )
                if len(rows) >= n:
                    break
    return pd.DataFrame(rows)


def load_reviews() -> pd.DataFrame:
    """Sammelt Rezensionen aus allen konfigurierten Kategorien."""
    frames = []
    for category in CATEGORIES:
        print(f"[data] Lade Kategorie '{category}' ...")
        df = _collect_category(category, SAMPLES_PER_CATEGORY)
        print(f"[data]   {len(df)} Rezensionen gesammelt.")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"[data] Gesamt: {len(data)} Rezensionen aus {len(CATEGORIES)} Kategorien.")
    return data


if __name__ == "__main__":
    df = load_reviews()
    print(df.head())
    print(df["rating"].value_counts().sort_index())
