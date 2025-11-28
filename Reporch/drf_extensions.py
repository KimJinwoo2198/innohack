from drf_spectacular.extensions import OpenApiAuthenticationExtension
from rest_framework.versioning import NamespaceVersioning
from rest_framework import exceptions


class CookieAuthScheme(OpenApiAuthenticationExtension):
    """OpenAPI security scheme for JWTAuthenticationFromCookie."""

    target_class = "Users.authentication.JWTAuthenticationFromCookie"
    name = "cookieAuth"

    def get_security_definition(self, auto_schema):  # noqa: D401
        return {
            "type": "apiKey",
            "in": "cookie",
            "name": "accessToken",
            "description": "Authentication via HttpOnly cookie",
        }


class DefaultingNamespaceVersioning(NamespaceVersioning):
    """NamespaceVersioning variant that falls back to default_version instead of raising NotFound.

    drf-spectacular schema 생성 시 request 객체가 없어서 `determine_version` 가
    NotFound 예외를 일으키는 문제를 방지한다.
    """

    def determine_version(self, request, *args, **kwargs):  # type: ignore[override]
        try:
            return super().determine_version(request, *args, **kwargs)
        except exceptions.NotFound:
            return self.default_version