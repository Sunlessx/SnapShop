# SnapShop: visual product search

Take a photo of a piece of clothing (or any product), send it to the API, and get back the most similar items from one or more shop catalogs: our own catalog, Shein, Temu, or any shop that gives out a product feed.

## How it works

```
Client → POST /search (image + optional list of shops)
            │
            ▼
        FastAPI (port 8100)
            │
   ┌────────┼─────────────────┐
   ▼        ▼                 ▼
 CLIP    FAISS          product sources
(local) (in memory)   Typesense / CSV / JSON feeds
```

1. **CLIP** (ViT-B/32, runs on the machine, no paid API) turns the photo into a vector of 512 numbers.
2. **FAISS** (`IndexFlatIP`) compares that vector with every product vector and keeps the closest ones (cosine similarity, exact search).
3. The API returns those products with their name, image, price, shop and link.

The product vectors are computed ahead of time by `scripts/index_catalog.py`, which reads every source listed in `sources.json`.

## Why it's built this way

- **CLIP runs locally**: embedding images is the expensive step, so paying an API per image doesn't make sense. Redis caches the vectors so the same photo is never embedded twice.
- **Flat index, no vector DB**: under ~100k products an exact search is already sub-millisecond, and there's nothing to tune.
- **Sources are read-only**: the service never writes to Typesense or the shops' feeds.
- **Safe rebuilds**: the indexer writes to `.tmp` files and swaps them in at the end, so a crash halfway never breaks the index being served.

## Multiple shops (sources)

Copy `sources.example.json` to `sources.json` and list the datasets you want:

```json
[
  { "name": "main",  "type": "typesense" },
  { "name": "shein", "type": "feed", "location": "data/feeds/shein.csv" },
  { "name": "temu",  "type": "feed", "location": "https://example.com/temu_products.json",
    "fields": { "img": "thumb_url", "url": "goods_link" } }
]
```

- `typesense`: the catalog configured in `.env`.
- `feed`: a CSV or JSON file, either local or at a URL. Common column names (`title`, `image_link`, `price`, `sale_price`, `link`...) are detected automatically, and `fields` lets you map anything unusual.

Where do Shein/Temu feeds come from? Their **affiliate programs** give out product feeds and APIs. Scraping their websites directly is against their terms and gets blocked quickly, so this project doesn't do it.

Without a `sources.json`, the project behaves like before and only indexes the Typesense catalog.

## Live web search (Shein and others)

Send `web=true` with a search and the API also looks on Google Shopping, through [SerpAPI](https://serpapi.com) (free plan: 250 searches/month). Put your key in `.env` as `SERPAPI_KEY=...`.

1. CLIP describes the photo in words by comparing it with a list of clothing types and colours (e.g. "navy blue jeans").
2. That text is searched on Google Shopping, once per shop in `web_shops` (default `shein`, 1 credit each; empty = any shop, 1 credit).
3. The result thumbnails are downloaded and re-ranked with CLIP, so the items that really look like the photo come first.

The photo itself never leaves the machine, only the text query. Web items are mixed into `results` with the local ones, all sorted by score (`source` says where each comes from), and `web_query` shows the description used. Queries are cached for 24 h, so the same search twice costs one credit. Temu barely lists on Google Shopping, so in practice this finds Shein and many other shops but rarely Temu.

## API

### `GET /health`
```json
{ "status": "ok", "indexed_products": 21340 }
```

### `GET /sources`
How many products are indexed per shop.
```json
{ "sources": { "main": 20000, "shein": 1200, "temu": 140 } }
```

### `POST /search`
Multipart form: `image` (required), `top_k` (default 10, max `SEARCH_TOP_K`), `sources` (optional, comma separated; empty = all shops).

```bash
curl -X POST http://localhost:8100/search \
  -F "image=@/path/to/photo.jpg" \
  -F "top_k=5" \
  -F "sources=shein,temu" | jq
```

```json
{
  "results": [
    {
      "product_id": "136904",
      "name": "Blue Strappy Heels",
      "img": "https://cdn.example.com/...",
      "selling_price": 999.0,
      "discount_price": 799.0,
      "score": 0.94,
      "source": "shein",
      "url": "https://..."
    }
  ],
  "count": 5
}
```

Accepted formats: JPEG, PNG, WebP, max 10 MB. The file type is checked from its first bytes, not its extension.

## Getting started

```bash
git clone https://github.com/Sunlessx/SnapShop.git
cd SnapShop
python -m venv venv
venv/bin/pip install -e .          # Windows: venv\Scripts\pip install -e .

cp .env.example .env               # fill in the Typesense host + read-only key
cp sources.example.json sources.json   # optional, for several shops

python scripts/index_catalog.py --limit 100     # small test index
uvicorn visual_search.main:app --host 127.0.0.1 --port 8100
```

### Indexing

| Command | What it does |
|---|---|
| `python scripts/index_catalog.py --limit 20 --dry-run` | Quick check: embeds 20 products per source, saves nothing |
| `python scripts/index_catalog.py --limit 100` | Small index to try the API |
| `python scripts/index_catalog.py --source shein --source temu` | Only some sources |
| `python scripts/index_catalog.py` | Everything (about 30–90 min on CPU for 20k products) |

**Test data:** `python scripts/make_test_feed.py` (needs `pip install pyarrow`) downloads 1000 clothes of 64 types and 44 colours from a public HuggingFace dataset into `data/feeds/fashion.json` + `data/images/`. Add `{ "name": "fashion", "type": "feed", "location": "data/feeds/fashion.json" }` to `sources.json`, index, and the images are viewable at `http://localhost:8100/images/<id>.jpg`.

The index is saved to `data/embeddings.index` and the product info to `data/metadata.pkl` (row *i* of one matches row *i* of the other). Restart the API to load a new index. Each run rebuilds the full index, so `--source` replaces the index with only those shops.

## Docker

```bash
docker build -t snapshop .
docker run -p 8100:8100 --env-file .env -v $(pwd)/data:/app/data snapshop
```

The CLIP weights are downloaded during the build so the container starts fast. Mount `sources.json` too if you use several shops.

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `TYPESENSE_HOST` | ; | Typesense host (required for the `typesense` source) |
| `TYPESENSE_PORT` | `443` | Typesense port |
| `TYPESENSE_PROTOCOL` | `https` | Typesense protocol |
| `TYPESENSE_API_KEY` | ; | Read-only Typesense key |
| `TYPESENSE_COLLECTION` | `products` | Collection name |
| `SOURCES_FILE` | `sources.json` | List of shops to index |
| `INDEX_PATH` | `data/embeddings.index` | FAISS index file |
| `METADATA_PATH` | `data/metadata.pkl` | Product info file |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | HuggingFace model |
| `CLIP_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SEARCH_TOP_K` | `20` | Max results per search |
| `MAX_UPLOAD_BYTES` | `10485760` | Max upload size (10 MB) |
| `REDIS_URL` | `redis://localhost:6379` | Redis for caching (optional) |
| `REDIS_KEY_PREFIX` | `vs:` | Prefix for every Redis key |
| `SERPAPI_KEY` | — | SerpAPI key for the live web search (optional) |
| `WEB_COUNTRY` | `us` | Google Shopping country for web results |

Redis is optional: if it can't be reached, caching turns itself off and search still works.

## Ideas for later

- Switch to `IndexIDMap(IndexFlatIP)` to add or remove single products without a full rebuild.
- Index each shop into its own file so one shop can be refreshed alone.
