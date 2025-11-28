from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import exceptions
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class JWTAuthenticationFromCookie(JWTAuthentication):
    """
    JWT authentication that reads the access token from a secure HttpOnly cookie.
    """
    def authenticate(self, request):
        raw_token = request.COOKIES.get(settings.JWT_ACCESS_COOKIE_NAME)
        if raw_token:
            try:
                validated_token = self.get_validated_token(raw_token)
                user = self.get_user(validated_token)
                return (user, validated_token)
            except exceptions.AuthenticationFailed as e:
                logger.debug(f"[JWTAuth] cookie token failed: {e}; falling back to header")
            except Exception as e:
                logger.debug(f"[JWTAuth] cookie token exception: {type(e).__name__} - {e}; falling back to header")
        else:
            logger.debug("[JWTAuth] no cookie raw_token found; falling back to header")
        header_auth = super().authenticate(request)
        return header_auth 