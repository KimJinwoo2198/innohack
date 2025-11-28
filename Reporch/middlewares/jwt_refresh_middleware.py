from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
from django.utils import timezone
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
import base64, json
from django.core.cache import cache
import hashlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import orjson
except ImportError:
    orjson = None

User = get_user_model()

class JWTRefreshMiddleware(MiddlewareMixin):
    """
    요청 시 엑세스 토큰이 만료되었으면 자동으로 리프레시 토큰을 사용해 토큰을 갱신하고,
    응답에 새로운 토큰을 쿠키로 설정하는 미들웨어입니다.
    """

    def _is_token_valid(self, token):
        """페이로드만 디코딩하여 만료 여부를 빠르게 확인"""
        # 간략 토큰 유효성 확인: 페이로드만 디코딩
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return False
            payload = parts[1]
            payload_bytes = self._decode_payload_bytes(payload)
            data = orjson.loads(payload_bytes) if orjson else json.loads(payload_bytes)
            exp = data.get('exp', 0)
            now_ts = int(timezone.now().timestamp())
            return exp > now_ts
        except Exception:
            return False

    def process_request(self, request):
        access_name = settings.JWT_ACCESS_COOKIE_NAME
        refresh_name = settings.JWT_REFRESH_COOKIE_NAME
        access_token = request.COOKIES.get(access_name)
        refresh_token = request.COOKIES.get(refresh_name)
        if not access_token and not refresh_token:
            return None
        if access_token and self._is_token_valid(access_token):
            return None
        if not refresh_token:
            return None
        key_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        cache_key = f'jwt_refresh:{key_hash}'
        cached = cache.get(cache_key)
        if cached:
            new_access_str = cached['access']
            new_refresh_str = cached['refresh']
        else:
            try:
                rt = RefreshToken(refresh_token)
                rotate = settings.SIMPLE_JWT.get('ROTATE_REFRESH_TOKENS', False)
                if rotate:
                    user_id = rt[settings.SIMPLE_JWT['USER_ID_CLAIM']]
                    try:
                        user = User.objects.get(pk=user_id)
                    except User.DoesNotExist:
                        return None
                    new_rt = RefreshToken.for_user(user)
                else:
                    new_rt = rt
                new_access_str = str(new_rt.access_token)
                new_refresh_str = str(new_rt)
                cache.set(cache_key, {'access': new_access_str, 'refresh': new_refresh_str}, timeout=60)
            except TokenError:
                return None

        request.META['HTTP_AUTHORIZATION'] = f'Bearer {new_access_str}'
        request._jwt_refresh = {'access': new_access_str, 'refresh': new_refresh_str}
        request.COOKIES[access_name] = new_access_str
        request.COOKIES[refresh_name] = new_refresh_str

    @lru_cache(maxsize=1024)
    def _decode_payload_bytes(self, payload_str):
        rem = len(payload_str) % 4
        if rem:
            payload_str += '=' * (4 - rem)
        return base64.urlsafe_b64decode(payload_str)

    def process_response(self, request, response):
        data = getattr(request, '_jwt_refresh', None)
        if data:
            response.set_cookie(
                key=settings.JWT_ACCESS_COOKIE_NAME,
                value=data['access'],
                httponly=settings.JWT_COOKIE_HTTPONLY,
                secure=settings.JWT_COOKIE_SECURE,
                samesite=settings.JWT_COOKIE_SAMESITE,
                domain=settings.JWT_COOKIE_DOMAIN,
                # max_age=int(settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME'].total_seconds()),
                path=settings.JWT_COOKIE_PATH,
            )
            response.set_cookie(
                key=settings.JWT_REFRESH_COOKIE_NAME,
                value=data['refresh'],
                httponly=settings.JWT_COOKIE_HTTPONLY,
                secure=settings.JWT_COOKIE_SECURE,
                samesite=settings.JWT_COOKIE_SAMESITE,
                domain=settings.JWT_COOKIE_DOMAIN,
                # max_age=int(settings.SIMPLE_JWT['REFRESH_TOKEN_LIFETIME'].total_seconds()),
                path=settings.JWT_COOKIE_PATH,
            )
        return response 