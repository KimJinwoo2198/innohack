from django.urls import re_path

from vision.consumers import FoodChatConsumer

websocket_urlpatterns = [
    re_path(r"^ws/vision/foods/chat/$", FoodChatConsumer.as_asgi()),
]


