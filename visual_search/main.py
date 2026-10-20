from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import settings
from .embedder import embed_image_bytes
from .index_store import IndexStore
from .redis_cache import cache_get, cache_set, hash_bytes
from .schemas import ProductMatch, SearchResponse
from .web_search import search_web

SEARCH_TTL = 3600  # 1h, the indexer clears these anyway after a rebuild

store = IndexStore(settings.index_path, settings.metadata_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the index once at startup, if it's not built yet /search returns 503
    if store.index_path.exists() and store.metadata_path.exists():
        store.load()
    yield


app = FastAPI(title="SnapShop", version="1.1.0", lifespan=lifespan)

# images downloaded by scripts/make_test_feed.py, so results can be opened in the browser
IMAGES_DIR = Path("data/images")
if IMAGES_DIR.is_dir():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
# first bytes of each format, so we don't trust the file extension
IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",
}


def detect_image_type(data: bytes) -> str | None:
    for sig, mime in IMAGE_SIGNATURES.items():
        if data[:len(sig)] == sig:
            # RIFF is also used by wav/avi, webp has "WEBP" at byte 8
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return None


@app.get("/health")
def health():
    return {"status": "ok", "indexed_products": store.total}


@app.get("/sources")
def list_sources():
    # how many products we have from each dataset
    return {"sources": store.source_counts()}


@app.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    sources: str | None = Form(None),  # e.g. "shein,temu", empty = search everything
    web: bool = Form(False),  # also search google shopping (costs serpapi credits)
    web_shops: str = Form("shein"),  # 1 credit per shop, empty = any shop for 1 credit
):
    data = await image.read()

    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    mime = detect_image_type(data) or image.content_type
    if mime not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, WebP accepted")

    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Index not ready — run scripts/index_catalog.py first")

    k = max(1, min(top_k, settings.search_top_k))
    wanted = {s.strip() for s in sources.split(",") if s.strip()} if sources else None
    if wanted:
        # better a clear error than an empty list when there's a typo in the shop name
        unknown = wanted - store.source_counts().keys()
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source(s): {', '.join(sorted(unknown))}. "
                       f"Available: {', '.join(sorted(store.source_counts()))}. Leave empty to search all.",
            )

    # same image + same k + same shops -> same answer, so cache on all three
    embed_h = hash_bytes(data)
    src_key = ",".join(sorted(wanted)) if wanted else "all"
    search_key = f"img:search:{embed_h}:{k}:{src_key}"
    hits = cache_get(search_key)
    if hits is None:
        embedding = embed_image_bytes(data)
        hits = store.search(embedding, top_k=k, sources=wanted)
        cache_set(search_key, hits, SEARCH_TTL)

    all_hits = list(hits)
    web_query = web_error = None

    if web:
        shops = [s.strip() for s in web_shops.split(",") if s.strip()]
        try:
            # blocking work (CLIP + http calls), keep it off the event loop
            web_query, web_hits = await run_in_threadpool(
                lambda: search_web(embed_image_bytes(data), shops, k)
            )
            all_hits += web_hits
        except Exception as e:
            # web search is a bonus, the local results still go back
            web_error = str(e)

    # one list for everything: both scores are CLIP similarity to the same photo,
    # so local and web items can be compared directly. no score = last
    all_hits.sort(key=lambda h: h.get("score") if h.get("score") is not None else -1, reverse=True)
    results = [_to_match(h) for h in all_hits[:k]]
    return SearchResponse(results=results, count=len(results), web_query=web_query, web_error=web_error)


def _to_match(h: dict) -> ProductMatch:
    return ProductMatch(
        product_id=str(h.get("product_id", h.get("id", ""))),
        name=h.get("name", ""),
        img=h.get("img", ""),
        selling_price=float(h.get("selling_price") or 0),
        discount_price=float(h["discount_price"]) if h.get("discount_price") else None,
        score=h.get("score"),
        source=h.get("source", "main"),
        url=h.get("url"),
    )
