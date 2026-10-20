from pydantic import BaseModel
from typing import Optional


class ProductMatch(BaseModel):
    product_id: str
    name: str
    img: str
    selling_price: float
    discount_price: Optional[float] = None
    score: Optional[float] = None
    source: str = "main"  # which dataset it came from (main, shein, temu...)
    url: Optional[str] = None  # link to the product page when the feed has one


class SearchResponse(BaseModel):
    results: list[ProductMatch]
    count: int
    # only filled when web=true (web items are mixed into results, see "source")
    web_query: Optional[str] = None  # what CLIP thinks the photo is, e.g. "red dress"
    web_error: Optional[str] = None
