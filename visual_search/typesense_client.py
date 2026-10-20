from typing import Generator

import typesense

from .config import settings


def get_client() -> typesense.Client:
    return typesense.Client({
        "nodes": [{
            "host": settings.typesense_host,
            "port": settings.typesense_port,
            "protocol": settings.typesense_protocol,
        }],
        "api_key": settings.typesense_api_key,
        "connection_timeout_seconds": 10,
    })


def fetch_all_products(client: typesense.Client) -> Generator[dict, None, None]:
    # read-only, we page through the whole collection 250 at a time (typesense max)
    page = 1
    per_page = 250
    while True:
        result = client.collections[settings.typesense_collection].documents.search({
            "q": "*",
            "per_page": per_page,
            "page": page,
            "filter_by": "active:=1",
            "include_fields": "id,product_id,name,img,selling_price,discount_price",
        })
        hits = result.get("hits", [])
        if not hits:
            break
        for hit in hits:
            yield hit["document"]
        if len(hits) < per_page:  # last page
            break
        page += 1
