"""Lightweight Redis cache helper. Falls through gracefully if Redis is unavailable."""
import hashlib
import json
import logging
from typing import Any, Optional

try:
    import redis as _redis_pkg
except ImportError:
    _redis_pkg = None

from .config import settings

log = logging.getLogger(__name__)

REDIS_URL = settings.redis_url
PREFIX = settings.redis_key_prefix  # so we don't clash with other apps on the same redis

_client = None
_disabled = False


def _get_client():
    # connect lazily, and if redis is down just turn caching off for good
    global _client, _disabled
    if _disabled or _redis_pkg is None:
        return None
    if _client is None:
        try:
            _client = _redis_pkg.from_url(
                REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,
            )
            _client.ping()
            log.info("Redis cache connected: %s", REDIS_URL)
        except Exception as e:
            log.warning("Redis cache disabled: %s", e)
            _disabled = True
            _client = None
    return _client


def _k(key: str) -> str:
    return key if key.startswith(PREFIX) else PREFIX + key


def cache_get(key: str) -> Optional[Any]:
    c = _get_client()
    if c is None:
        return None
    try:
        raw = c.get(_k(key))
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("cache_get failed (%s): %s", key, e)
        return None


def cache_set(key: str, value: Any, ttl_sec: int) -> None:
    c = _get_client()
    if c is None:
        return
    try:
        c.set(_k(key), json.dumps(value), ex=ttl_sec)
    except Exception as e:
        log.warning("cache_set failed (%s): %s", key, e)


def cache_get_bytes(key: str) -> Optional[bytes]:
    c = _get_client()
    if c is None:
        return None
    try:
        return c.get(_k(key))
    except Exception:
        return None


def cache_set_bytes(key: str, value: bytes, ttl_sec: int) -> None:
    c = _get_client()
    if c is None:
        return
    try:
        c.set(_k(key), value, ex=ttl_sec)
    except Exception:
        pass


def cache_del(pattern: str) -> int:
    # SCAN instead of KEYS so we don't block redis on big keyspaces
    c = _get_client()
    if c is None:
        return 0
    full = _k(pattern)
    try:
        if "*" not in full:
            return c.delete(full)
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = c.scan(cursor=cursor, match=full, count=200)
            if keys:
                deleted += c.delete(*keys)
            if cursor == 0:
                break
        return deleted
    except Exception as e:
        log.warning("cache_del failed (%s): %s", pattern, e)
        return 0


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
