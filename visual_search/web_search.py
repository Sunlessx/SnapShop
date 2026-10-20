"""
Live search on the internet (Google Shopping through SerpAPI).

Google Lens would need a public link to the photo, and our server runs locally,
so instead:
  1. CLIP "describes" the photo by comparing it with a list of clothing types
     and colours (the text half of CLIP, same model we already load)
  2. we search Google Shopping for that description, e.g. "red dress shein"
  3. we download the thumbnails of the results and re-rank them with CLIP
     so the ones that actually look like the photo come first

The photo never leaves this machine, only the text query does.
Each shop searched = 1 SerpAPI credit (free plan = 250/month), results are cached.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import httpx
import numpy as np

from .config import settings
from .embedder import embed_image_url, embed_texts
from .redis_cache import cache_get, cache_set

log = logging.getLogger(__name__)

SERP_URL = "https://serpapi.com/search.json"
SERP_TTL = 24 * 3600  # a day, shop listings don't change that fast
MAX_RERANK = 20  # thumbnails we download per shop, keeps it under a few seconds

TYPES = [
    "t-shirt", "shirt", "top", "blouse", "sweater", "hoodie", "sweatshirt", "jacket",
    "coat", "blazer", "dress", "skirt", "jeans", "trousers", "shorts", "leggings",
    "track pants", "jumpsuit", "suit", "kurta", "saree", "swimsuit", "bra", "lingerie",
    "socks", "sneakers", "sports shoes", "heels", "sandals", "flip flops", "boots",
    "flats", "formal shoes", "handbag", "backpack", "cap", "scarf", "belt",
]
COLOURS = [
    "black", "white", "grey", "navy blue", "blue", "light blue", "red", "maroon",
    "pink", "purple", "green", "olive", "yellow", "orange", "brown", "beige",
    "cream", "gold", "silver", "multicolour",
]

# no redis running? still avoid paying twice for the same query while the server is up
_memory_cache: dict[str, list] = {}


@lru_cache(maxsize=1)
def _label_vectors():
    # computed once, ~60 short sentences
    types = embed_texts([f"a photo of a {t}" for t in TYPES])
    colours = embed_texts([f"a photo of {c} clothes" for c in COLOURS])
    return types, colours


def describe(image_vec: np.ndarray) -> str:
    types, colours = _label_vectors()
    t = TYPES[int(np.argmax(types @ image_vec))]
    c = COLOURS[int(np.argmax(colours @ image_vec))]
    return f"{c} {t}"


def _shopping_results(query: str) -> list[dict]:
    key = f"web:serp:{settings.web_country}:{query}"
    if key in _memory_cache:
        return _memory_cache[key]
    cached = cache_get(key)
    if cached is not None:
        return cached

    resp = httpx.get(SERP_URL, timeout=30, params={
        "engine": "google_shopping",
        "q": query,
        "gl": settings.web_country,
        "hl": "en",
        "api_key": settings.serpapi_key,
    })
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    results = data.get("shopping_results", [])
    _memory_cache[key] = results
    cache_set(key, results, SERP_TTL)
    return results


def _to_product(r: dict) -> dict:
    price = r.get("extracted_price")
    old = r.get("extracted_old_price")
    return {
        "product_id": str(r.get("product_id", "")),
        "name": r.get("title", ""),
        "img": r.get("thumbnail", ""),
        # when there's an old price, the shop is showing a discount
        "selling_price": float(old if old else price or 0),
        "discount_price": float(price) if old and price else None,
        "source": r.get("source", "web"),
        "url": r.get("product_link") or r.get("link"),
    }


def _score(item: dict, query_vec: np.ndarray) -> dict:
    try:
        item["score"] = float(embed_image_url(item["img"]) @ query_vec)
    except Exception:
        item["score"] = None  # thumbnail didn't load, goes to the end
    return item


def search_web(image_vec: np.ndarray, shops: list[str], top_k: int) -> tuple[str, list[dict]]:
    if not settings.serpapi_key:
        raise RuntimeError("SERPAPI_KEY is not set in .env")

    description = describe(image_vec)
    items = []
    # one search per shop, or one general search if no shop given
    for shop in shops or [""]:
        query = f"{description} {shop}".strip()
        found = [_to_product(r) for r in _shopping_results(query)]
        if shop:
            found = [f for f in found if shop.lower() in f["source"].lower()]
        items += found[:MAX_RERANK]

    # same product can show up twice with different links, so compare name + price
    seen, unique = set(), []
    for it in items:
        key = (it["name"].lower(), it["selling_price"])
        if key not in seen:
            seen.add(key)
            unique.append(it)

    with ThreadPoolExecutor(max_workers=8) as pool:
        scored = list(pool.map(lambda it: _score(it, image_vec), unique))
    scored.sort(key=lambda it: it["score"] if it["score"] is not None else -1, reverse=True)
    return description, scored[:top_k]
