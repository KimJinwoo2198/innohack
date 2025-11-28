from django.contrib import admin as django_admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from .views import health

urlpatterns = [
   path('admin/', django_admin.site.urls),
   path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
   path('api/schema/swagger/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
   path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
   path('api/v1/', include('Users.urls')),
   path('api/v1/vision/', include('vision.urls')),
   path('health/', health, name='health'),
   path('i18n/', include('django.conf.urls.i18n')),
]

# 개발 환경에서 미디어 파일 제공
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

handler404 = 'Reporch.views.empty_404'