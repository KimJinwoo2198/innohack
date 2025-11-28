from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags
from django.conf import settings
from .models import CustomUser
from django.conf import settings as _settings
from django.utils import timezone as _tz
import json
from django.template.loader import render_to_string

logger = get_task_logger(__name__)

@shared_task(bind=True, max_retries=2)
def warm_profile_cache(self, user_id: int):
    try:
        user = CustomUser.objects.only(
            'id', 'username', 'email', 'is_superuser', 'last_login', 'profile_image', 'has_active_subscription'
        ).get(pk=user_id)
    except Exception as exc:
        logger.exception("warm_profile_cache: user fetch failed: %s", exc)
        return False

    try:
        version_key = f"profile_version:{user_id}"
        version = cache.get(version_key)
        if version is None:
            version = 1
            cache.set(version_key, version)

        cdn_url = None
        name = getattr(getattr(user, 'profile_image', None), 'name', None)
        if name:
            cdn_base = getattr(_settings, 'MEDIA_URL', None)
            if cdn_base:
                if not cdn_base.endswith('/'):
                    cdn_base = cdn_base + '/'
                cdn_url = f"{cdn_base}{name}"

        data = {
            "status": "success",
            "code": 200,
            "message": "사용자 정보를 성공적으로 반환했습니다.",
            "data": {
                "id": user_id,
                "username": user.username,
                "email": user.email,
                "is_superuser": user.is_superuser,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "profile_image_url": cdn_url,
                "billing": {
                    "isActive": bool(getattr(user, 'has_active_subscription', False)),
                    "planName": None,
                },
            },
            "meta": {"timestamp": _tz.now().isoformat()}
        }
        try:
            import orjson as _orjson  # type: ignore
            body = _orjson.dumps(data)
        except Exception:
            import json as _json
            body = _json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        etag = f"W/\"{user_id}-{version}\""
        cache.set(f"profile_data:{user_id}:{version}", {"body_bytes": body, "etag": etag}, timeout=300)
        return True
    except Exception as exc:
        logger.exception("warm_profile_cache failed: %s", exc)
        return False

def send_email(subject, to_email, template_name, context):
    try:
        html_content = render_to_string(template_name, context)
        text_content = strip_tags(html_content)
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[to_email],
        )
        email.attach_alternative(html_content, "text/html")
        email.send()
        logger.info("Email sent to %s with subject '%s'", to_email, subject)
    except Exception as exc:
        logger.exception("Failed to send email to %s", to_email)
        raise

