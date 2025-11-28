import logging
from http.cookies import SimpleCookie
from typing import Dict, Optional
from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, AuthenticationFailed

logger = logging.getLogger(__name__)


class JWTAuthMiddleware(BaseMiddleware):
    """
    WebSocket용 JWT 인증 미들웨어.
    쿠키 우선으로 토큰을 추출하고, 없으면 Authorization 헤더나 쿼리 파라미터(token)를 사용한다.
    """

    def __init__(self, inner):
        super().__init__(inner)
        self.jwt_auth = JWTAuthentication()

    async def __call__(self, scope, receive, send):
        scope = dict(scope)
        cookies = self._parse_cookies(scope)
        scope['cookies'] = cookies
        scope['anonymous_id'] = cookies.get('anon_id', '')

        raw_token = self._extract_token(scope, cookies)
        scope['user'] = await self._resolve_user(raw_token)
        return await super().__call__(scope, receive, send)

    async def _resolve_user(self, raw_token: Optional[str]):
        if not raw_token:
            return AnonymousUser()
        try:
            return await self._get_user(raw_token)
        except (InvalidToken, AuthenticationFailed) as exc:
            logger.debug('JWTAuthMiddleware token rejected: %s', exc)
            return AnonymousUser()
        except Exception as exc:  # pragma: no cover - 방어적 로깅
            logger.exception('JWTAuthMiddleware unexpected error: %s', exc)
            return AnonymousUser()

    @database_sync_to_async
    def _get_user(self, raw_token: str):
        validated = self.jwt_auth.get_validated_token(raw_token)
        return self.jwt_auth.get_user(validated)

    def _parse_cookies(self, scope) -> Dict[str, str]:
        jar = SimpleCookie()
        cookies: Dict[str, str] = {}
        for header_name, header_value in scope.get('headers', []):
            if header_name == b'cookie':
                try:
                    jar.load(header_value.decode('utf-8'))
                except Exception:  # pragma: no cover - 방어
                    continue
        for key, morsel in jar.items():
            cookies[key] = morsel.value
        return cookies

    def _extract_token(self, scope, cookies: Dict[str, str]) -> Optional[str]:
        cookie_token = cookies.get(settings.JWT_ACCESS_COOKIE_NAME)
        if cookie_token:
            return cookie_token

        for header_name, header_value in scope.get('headers', []):
            if header_name == b'authorization':
                parts = header_value.decode('utf-8').split(' ', 1)
                if len(parts) == 2 and parts[0].lower() == 'bearer':
                    return parts[1].strip()

        query_string = scope.get('query_string', b'')
        if query_string:
            params = parse_qs(query_string.decode('utf-8'))
            token_param = params.get('token')
            if token_param:
                return token_param[0]
        return None

