from django.urls import path, include
from rest_framework.routers import SimpleRouter
from .views import (
    SignupViewSet,
    LoginViewSet,
    LogoutViewSet,
)

auth_router = SimpleRouter()
auth_router.register('', LoginViewSet, basename='login')
auth_router.register('', SignupViewSet, basename='signup')
auth_router.register('', LogoutViewSet, basename='logout')

urlpatterns = [
    path('auth/', include(auth_router.urls)),
]
