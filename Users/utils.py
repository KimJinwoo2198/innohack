from django.core.cache import cache
from django.conf import settings
try:
    from storages.backends.s3boto3 import S3Boto3Storage
except ImportError:
    S3Boto3Storage = None


def generate_presigned_get_url(key: str, expires: int = 3600) -> str | None:

    if not key:
        return None

    try:
        if S3Boto3Storage and hasattr(settings, 'AWS_STORAGE_BUCKET_NAME') and settings.AWS_STORAGE_BUCKET_NAME:
            cache_key = f"presign:{key}:{expires}"
            cached = cache.get(cache_key)
            if cached:
                return cached

            storage = S3Boto3Storage()
            client = storage.connection.meta.client
            bucket_name = getattr(storage, "bucket_name", settings.AWS_STORAGE_BUCKET_NAME)

            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": key},
                ExpiresIn=expires,
            )

            cache.set(cache_key, url, timeout=max(expires - 60, 60))
            return url
    except Exception:
        pass

    try:
        from django.core.files.storage import default_storage
        url = default_storage.url(key)
        return url
    except Exception:
        return None
