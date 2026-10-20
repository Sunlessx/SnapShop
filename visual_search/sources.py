"""
Product sources (datasets) the indexer can pull from.

Two kinds for now:
  - "typesense": our own catalog, same as before
  - "feed":      a CSV or JSON product feed, either a local file or a URL.
                 Shops like Shein / Temu give these out through their affiliate
                 programs, so that's the way to get their products in
                 (scraping their sites directly breaks their ToS and gets blocked fast).

Every product that comes out of here has the same shape, plus a "source"
field so we know which shop it came from.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx

from .config import settings
from .typesense_client import fetch_all_products, get_client

log = logging.getLogger(__name__)

# names we accept for each field in a feed, first match wins.
# feeds are never consistent so this saves writing a mapping every time
DEFAULT_FIELD_NAMES = {
    "product_id": ["product_id", "id", "sku", "goods_id"],
    "name": ["name", "title", "product_name", "goods_name"],
    "img": ["img", "image", "image_url", "image_link", "main_image"],
    "selling_price": ["selling_price", "price", "retail_price", "original_price"],
    "discount_price": ["discount_price", "sale_price", "special_price"],
    "url": ["url", "link", "product_url", "detail_url"],
}


@dataclass
class Source:
    name: str
    type: str = "typesense"
    location: str = ""  # path or http(s) url, only for feeds
    fields: dict[str, str] = field(default_factory=dict)  # optional custom mapping


def load_sources(path: str | None = None) -> list[Source]:
    p = Path(path or settings.sources_file)
    if not p.exists():
        return [Source(name="main", type="typesense")]

    raw = json.loads(p.read_text(encoding="utf-8"))
    sources = [Source(**s) for s in raw]
    names = [s.name for s in sources]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate source names in {p}: {names}")
    return sources


def iter_products(source: Source) -> Iterator[dict]:
    if source.type == "typesense":
        for doc in fetch_all_products(get_client()):
            yield _normalize(doc, {}, source.name)
    elif source.type == "feed":
        for row in _read_feed(source.location):
            yield _normalize(row, source.fields, source.name)
    else:
        raise ValueError(f"unknown source type '{source.type}' for {source.name}")


def _read_feed(location: str) -> list[dict]:
    if location.startswith(("http://", "https://")):
        resp = httpx.get(location, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
    else:
        text = Path(location).read_text(encoding="utf-8")

    # guess format from the extension, json otherwise if it looks like json
    loc = location.lower().split("?")[0]
    if loc.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(text)))

    data = json.loads(text)
    if isinstance(data, dict):
        # some feeds wrap the list, e.g. {"products": [...]}
        data = data.get("products") or data.get("items") or data.get("data") or []
    return data


def _pick(row: dict, key: str, custom: dict[str, str]):
    if key in custom:
        return row.get(custom[key])
    for name in DEFAULT_FIELD_NAMES[key]:
        if row.get(name) not in (None, ""):
            return row[name]
    return None


def _to_float(v) -> float | None:
    if v in (None, ""):
        return None
    try:
        # prices in feeds sometimes come as "12.99 USD" or "$12.99"
        return float(str(v).replace("$", "").replace("€", "").split()[0])
    except ValueError:
        return None


def _normalize(row: dict, custom: dict[str, str], source_name: str) -> dict:
    pid = _pick(row, "product_id", custom)
    return {
        "id": str(row.get("id", pid or "")),
        "product_id": str(pid or ""),
        "name": _pick(row, "name", custom) or "",
        "img": _pick(row, "img", custom) or "",
        "selling_price": _to_float(_pick(row, "selling_price", custom)) or 0.0,
        "discount_price": _to_float(_pick(row, "discount_price", custom)),
        "url": _pick(row, "url", custom),
        "source": source_name,
    }
