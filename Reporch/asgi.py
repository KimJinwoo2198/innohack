import importlib
import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Reporch.settings")

django_asgi_app = get_asgi_application()

try:
    civil_module = importlib.import_module("CivilAffairs.routing")  # noqa: E402
    civil_websocket_urls = getattr(civil_module, "websocket_urlpatterns", [])
except ModuleNotFoundError:
    civil_websocket_urls = []

from Users.ws_auth import JWTAuthMiddleware  # noqa: E402  pylint: disable=wrong-import-position

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": JWTAuthMiddleware(URLRouter(civil_websocket_urls)),
    }
)