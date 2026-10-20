#!/usr/bin/env python3
"""
Builds a test feed of clothes from the public "fashion product images" dataset
on HuggingFace (ashraq/fashion-product-images-small).

Downloads one parquet file of the dataset (~136 MB, ~22k products, images inside),
picks N clothes spread over as many types (shirts, jeans, dresses...) and colours
as possible, saves the images to data/images/ and writes data/feeds/fashion.json.
Then add this to sources.json:
    { "name": "fashion", "type": "feed", "location": "data/feeds/fashion.json" }

Needs pyarrow (only for this script):  pip install pyarrow

Usage:
    python scripts/make_test_feed.py            # 1000 items
    python scripts/make_test_feed.py --count 200
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

PARQUET_URL = (
    "https://huggingface.co/datasets/ashraq/fashion-product-images-small/resolve/main/"
    "data/train-00000-of-00002-6cff4c59f91661c3.parquet"
)
PARQUET = Path("data/fashion-small.parquet")
IMG_DIR = Path("data/images")
FEED = Path("data/feeds/fashion.json")


def download_parquet() -> None:
    if PARQUET.exists():
        return
    print("downloading dataset (~136 MB, only the first time)...")
    tmp = PARQUET.with_suffix(".tmp")
    with httpx.stream("GET", PARQUET_URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.replace(PARQUET)


def load_clothes() -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(PARQUET).to_pylist()
    # only real clothes + shoes, no watches/perfume/etc
    clothes = [r for r in rows if r["masterCategory"] in ("Apparel", "Footwear")]
    print(f"{len(clothes)} clothes out of {len(rows)} products")
    return clothes


def pick_balanced(rows: list[dict], count: int) -> list[dict]:
    # round robin over (type, colour) groups so every combo gets a turn
    groups = defaultdict(list)
    for r in rows:
        groups[(r["articleType"], r["baseColour"])].append(r)
    for g in groups.values():
        random.shuffle(g)
    keys = list(groups)
    random.shuffle(keys)

    picked = []
    while len(picked) < count and keys:
        for k in list(keys):
            if groups[k]:
                picked.append(groups[k].pop())
                if len(picked) == count:
                    break
            else:
                keys.remove(k)
    return picked


def save(row: dict) -> dict:
    path = IMG_DIR / f"{row['id']}.jpg"
    path.write_bytes(row["image"]["bytes"])
    return {
        "id": str(row["id"]),
        "name": row["productDisplayName"],
        "img": path.as_posix(),
        "url": f"/images/{path.name}",  # served by the API so you can open it
        "type": row["articleType"],
        "colour": row["baseColour"],
        "gender": row["gender"],
    }


def main(count: int, seed: int) -> None:
    random.seed(seed)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    FEED.parent.mkdir(parents=True, exist_ok=True)

    download_parquet()
    picked = pick_balanced(load_clothes(), count)
    items = [save(r) for r in picked]

    FEED.write_text(json.dumps(items, indent=1), encoding="utf-8")
    types = {i["type"] for i in items}
    colours = {i["colour"] for i in items}
    print(f"wrote {len(items)} items to {FEED} ({len(types)} types, {len(colours)} colours)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="same seed = same items")
    args = parser.parse_args()
    main(args.count, args.seed)
