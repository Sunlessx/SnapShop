import io
from functools import lru_cache

import numpy as np
from PIL import Image

from .config import settings
from .redis_cache import cache_get_bytes, cache_set_bytes, hash_bytes

EMBED_TTL = 30 * 86400  # 30 days


# only load CLIP once (and only when we actually need it, it's ~600MB)
@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    model = CLIPVisionModelWithProjection.from_pretrained(settings.clip_model_name)
    processor = CLIPImageProcessor.from_pretrained(settings.clip_model_name)
    model.eval()
    model.to(torch.device(settings.clip_device))
    return model, processor


@lru_cache(maxsize=1)
def _load_text_model():
    # the text half of CLIP, only used by the web search to describe photos in words
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    model = CLIPTextModelWithProjection.from_pretrained(settings.clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(settings.clip_model_name)
    model.eval()
    model.to(torch.device(settings.clip_device))
    return model, tokenizer


def embed_texts(texts: list[str]) -> np.ndarray:
    import torch

    model, tokenizer = _load_text_model()
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        features = model(**inputs).text_embeds  # shape: (len(texts), 512)
        # same normalization as images, so text and image vectors can be compared directly
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def embed_image_bytes(image_bytes: bytes) -> np.ndarray:
    import torch

    # same picture -> same vector, no point running the model twice
    h = hash_bytes(image_bytes)
    cache_key = f"img:embed:{h}"
    cached = cache_get_bytes(cache_key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32).copy()

    model, processor = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.image_embeds  # shape: (1, 512)
        # normalize so inner product in faiss gives cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
    vec = features[0].cpu().numpy().astype(np.float32)
    cache_set_bytes(cache_key, vec.tobytes(), EMBED_TTL)
    return vec


def embed_image_url(url: str) -> np.ndarray:
    import httpx

    # feeds can also point to images on disk (e.g. data/images/123.jpg)
    if not url.startswith(("http://", "https://")):
        with open(url, "rb") as f:
            return embed_image_bytes(f.read())

    # some CDNs block the default httpx user agent
    response = httpx.get(url, timeout=15.0, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (visual-search indexer)"})
    response.raise_for_status()
    return embed_image_bytes(response.content)
