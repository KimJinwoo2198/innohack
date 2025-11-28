from __future__ import annotations

import hashlib
import logging
from typing import Callable, TypeVar

from django.core.cache import cache

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _hash_suffix(raw_suffix: str) -> str:
    return hashlib.sha256(raw_suffix.encode("utf-8")).hexdigest()


def build_cache_key(user_id: int, namespace: str, suffix: str) -> str:
    digest = _hash_suffix(suffix)
    return f"vision:{namespace}:{user_id}:{digest}"


def get_cache_value(key: str) -> T | None:
    return cache.get(key)


def set_cache_value(key: str, value: T, timeout: int) -> None:
    cache.set(key, value, timeout)


def get_or_set(key: str, timeout: int, producer: Callable[[], T]) -> T:
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        value = producer()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Cache producer failed for key=%s: %s", key, exc)
        raise
    cache.set(key, value, timeout)
    return value


