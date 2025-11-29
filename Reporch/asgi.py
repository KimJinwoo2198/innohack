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

try:
    vision_module = importlib.import_module("vision.routing")  # noqa: E402  pylint: disable=wrong-import-position
    vision_websocket_urls = getattr(vision_module, "websocket_urlpatterns", [])
except ModuleNotFoundError:
    vision_websocket_urls = []

from Users.ws_auth import JWTAuthMiddleware  # noqa: E402  pylint: disable=wrong-import-position

websocket_routes = [*civil_websocket_urls, *vision_websocket_urls]

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": JWTAuthMiddleware(URLRouter(websocket_routes)),
    }
)