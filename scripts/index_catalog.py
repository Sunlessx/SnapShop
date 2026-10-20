#!/usr/bin/env python3
"""
Indexer: goes through every source in sources.json (typesense catalog,
shein/temu feeds, ...), computes a CLIP embedding for each product image
and saves one FAISS index + the metadata to disk.

Usage:
    python scripts/index_catalog.py
    python scripts/index_catalog.py --limit 50              # testing, 50 per source
    python scripts/index_catalog.py --source shein          # only some sources
    python scripts/index_catalog.py --dry-run               # no files written
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# so we can import visual_search without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_search.config import settings
from visual_search.embedder import embed_image_url
from visual_search.index_store import IndexStore
from visual_search.sources import iter_products, load_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("indexer")


def run(limit: int = 0, dry_run: bool = False, only: list[str] | None = None) -> None:
    sources = load_sources()
    if only:
        sources = [s for s in sources if s.name in only]
        if not sources:
            log.error(f"None of {only} found in {settings.sources_file}")
            return

    embeddings = []
    metadata = []
    errors = 0

    for source in sources:
        log.info(f"Fetching products from '{source.name}' ({source.type})...")
        done = 0
        try:
            for i, doc in enumerate(iter_products(source)):
                if limit and i >= limit:
                    break

                img_url = doc["img"]
                if not img_url:
                    log.warning(f"[{source.name}] no image for product {doc['product_id']}, skipping")
                    errors += 1
                    continue

                try:
                    embeddings.append(embed_image_url(img_url))
                    metadata.append(doc)
                    done += 1
                    if done % 100 == 0:
                        log.info(f"[{source.name}] embedded {done} products (errors={errors})")
                    # small pause so we don't hammer the image CDN (not needed for local files)
                    if img_url.startswith("http"):
                        time.sleep(0.05)
                except Exception as e:
                    log.error(f"[{source.name}] failed on product {doc['product_id']}: {e}")
                    errors += 1
        except Exception as e:
            # one broken feed shouldn't kill the whole run
            log.error(f"Source '{source.name}' failed: {e}")
            continue

        log.info(f"[{source.name}] done: {done} products")

    if not embeddings:
        log.error("No embeddings generated — nothing to save")
        return

    log.info(f"Building FAISS index with {len(embeddings)} vectors...")
    arr = np.stack(embeddings)
    store = IndexStore(settings.index_path, settings.metadata_path)
    store.build(arr, metadata)

    if dry_run:
        log.info(f"Dry run — skipping save. Would have saved {len(embeddings)} vectors (errors={errors})")
    else:
        store.save()
        log.info(f"Saved index to {settings.index_path} and metadata to {settings.metadata_path} (errors={errors})")
        # old cached results point to the old index, clear them
        try:
            from visual_search.redis_cache import cache_del
            deleted = cache_del("img:search:*")
            log.info(f"Invalidated {deleted} cached search results")
        except Exception as e:
            log.warning(f"Cache invalidation skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max products per source (0 = all)")
    parser.add_argument("--source", action="append", help="Only index this source (can repeat)")
    parser.add_argument("--dry-run", action="store_true", help="Skip saving index files")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run, only=args.source)
