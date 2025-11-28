# type: ignore
from pathlib import Path
import os
import json
from datetime import timedelta
import environ

# ================================
# 환경 변수 설정
# ================================

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    DEBUG=(bool, False),
)

env_file = os.path.join(BASE_DIR, '.env')
if os.path.exists(env_file):
    environ.Env.read_env(env_file)

# ================================
# 기본 설정
# ================================

BASE_URL = env('BASE_URL', default='http://localhost:8000')
FRONTEND_URL = env('FRONTEND_URL', default='http://localhost:3000')
SECRET_KEY = env('SECRET_KEY', default='django-insecure-change-me-in-production')
DEBUG = env('DEBUG', default=False)
ALLOWED_HOSTS = env('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')
ROOT_URLCONF = 'Reporch.urls'
APPEND_SLASH = True

# ================================
# 애플리케이션 설정
# ================================

INSTALLED_APPS = [
    'jazzmin',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.postgres',
    'channels',
    'Users',
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',
    'corsheaders',
    'drf_spectacular',
    'Reporch.drf_extensions',
    'pgvector.django',
    'vision',
]

SITE_ID = 1

# ================================
# 미들웨어 설정
# ================================
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'Reporch.middlewares.api_minimal_middleware.APIMinimalMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',  # CSRF 보호 완전 비활성화
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'Reporch.middlewares.jwt_refresh_middleware.JWTRefreshMiddleware',
    'Reporch.middlewares.custom_locale_middleware.CustomLocaleMiddleware',
]

# ================================
# 데이터베이스 설정
# ================================

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('POSTGRES_DB', default='drf_template_db'),
        'USER': env('POSTGRES_USER', default='postgres'),
        'PASSWORD': env('POSTGRES_PASSWORD', default='postgres'),
        'HOST': env('POSTGRES_HOST', default='localhost'),
        'PORT': env.int('POSTGRES_PORT', default=5432),
        'CONN_MAX_AGE': env.int('CONN_MAX_AGE', default=600),
        'OPTIONS': {
            'connect_timeout': 10,
        },
    }
}
PGVECTOR_EXTENSION = env('PGVECTOR_EXTENSION', default='vector')
VECTOR_DIM = env.int('VECTOR_DIM', default=3072)

OPENAI_API_KEY = env('OPENAI_API_KEY', default='')
OPENAI_MODEL_NAME = env('OPENAI_MODEL_NAME', default='gpt-5.1')
OPENAI_MODEL_OPTIONS = json.loads(env('OPENAI_MODEL_OPTIONS', default='{"reasoning_effort":"low"}'))
OPENAI_EMBED_MODEL = env('OPENAI_EMBED_MODEL', default='text-embedding-3-large')
MAPBOX_TOKEN = env('MAPBOX_TOKEN', default='')

# ================================
# 인증 설정
# ================================

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
]

AUTH_USER_MODEL = 'Users.CustomUser'

# ================================
# REST 프레임워크 설정
# ================================

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'Users.authentication.JWTAuthenticationFromCookie',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'Users.renderers.ORJSONRenderer',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.LimitOffsetPagination',
    'PAGE_SIZE': 50,
    'EXCEPTION_HANDLER': 'Users.custom_exception_handler.custom_exception_handler',
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    'DEFAULT_PARSER_CLASSES': [
        'Users.parsers.ORJSONParser',
    ],
    'DEFAULT_VERSIONING_CLASS': 'Reporch.drf_extensions.DefaultingNamespaceVersioning',
    'DEFAULT_VERSION': 'v1',
    'ALLOWED_VERSIONS': ['v1'],
    'DATETIME_FORMAT': '%Y-%m-%dT%H:%M:%S.%fZ',
    'DEFAULT_METADATA_CLASS': None,
    'UNAUTHENTICATED_USER': None,
    'UNAUTHENTICATED_TOKEN': None,
}

# ================================
# 사용자 인증 백엔드 설정
# ================================

AUTHENTICATION_BACKENDS = [
    'Users.backends.EmailOrUsernameModelBackend',
]

# ================================
# SWAGGER 설정
# ================================

SPECTACULAR_SETTINGS = {
    'OAS_VERSION': '3.1.0',
    'TITLE': '고민원 API',
    'DESCRIPTION': '부산 민원·행정 AI 코파일럿을 위한 고민원 백엔드 API.',
    'VERSION': '1.0.0',
    'CONTACT': {
        'name': 'API Support',
    },
    'LICENSE': {
        'name': 'MIT License',
    },
    'SERVE_PERMISSIONS': ['rest_framework.permissions.AllowAny'],
    'SERVE_AUTHENTICATION': ['Users.authentication.JWTAuthenticationFromCookie'],
    'SWAGGER_UI_SETTINGS': {
        'deepLinking': True,
        'defaultModelRendering': 'example',
        'defaultModelsExpandDepth': 1,
        'defaultModelExpandDepth': 2,
        'persistAuthorization': True,
        'displayRequestDuration': True,
    },
    'REDOC_UI_SETTINGS': {
        'expandResponses': '200,201',
        'hideDownloadButton': False,
    },
    'COMPONENT_SPLIT_REQUEST': True,
    'COMPONENT_NO_READ_ONLY_REQUIRED': True,
    'SORT_OPERATIONS': True,
    'SECURITY': [
        {'cookieAuth': []},
    ],
    'COMPONENTS': {
        'securitySchemes': {
            'cookieAuth': {
                'type': 'apiKey',
                'in': 'cookie',
                'name': 'accessToken',
                'description': 'Authentication via HttpOnly cookie',
            },
        },
    },
    'DISABLE_ERRORS_AND_WARNINGS': True,
}

# ================================
# JWT 설정
# ================================

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=60),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': env('JWT_SIGNING_KEY', default=SECRET_KEY),
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
}

# ================================
# JWT 쿠키 설정
# ================================
JWT_ACCESS_COOKIE_NAME = env('JWT_ACCESS_COOKIE_NAME', default='accessToken')
JWT_REFRESH_COOKIE_NAME = env('JWT_REFRESH_COOKIE_NAME', default='refreshToken')
JWT_COOKIE_SECURE = env.bool('JWT_COOKIE_SECURE', default=False)  # 개발 환경에서는 False
JWT_COOKIE_HTTPONLY = True
JWT_COOKIE_SAMESITE = env('JWT_COOKIE_SAMESITE', default='Lax')
JWT_COOKIE_PATH = '/'
JWT_COOKIE_DOMAIN = env('JWT_COOKIE_DOMAIN', default=None)

# ================================
# CORS 설정
# ================================

CORS_ALLOW_ALL_ORIGINS = True  # 모든 도메인 허용
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_HEADERS = True  # 모든 헤더 허용
CORS_ALLOW_ALL_METHODS = True  # 모든 HTTP 메서드 허용
CORS_ALLOWED_ORIGIN_REGEXES = [
    r".*",  # 모든 패턴 허용
]

# ================================
# CSRF 설정 (완전 비활성화)
# ================================

# CSRF 미들웨어가 비활성화되어 있지만, 추가 보안을 위한 설정
CSRF_COOKIE_SECURE = False  # HTTP에서도 작동
CSRF_COOKIE_HTTPONLY = False  # JavaScript에서 접근 가능
CSRF_USE_SESSIONS = False  # 세션 기반 CSRF 비활성화
CSRF_COOKIE_SAMESITE = None  # SameSite 제한 제거
SECURE_CROSS_ORIGIN_OPENER_POLICY = None
# CSRF_TRUSTED_ORIGINS 설정 제거 (미들웨어 비활성화로 불필요)

# ================================
# 캐시 설정
# ================================

CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': f'redis://{env("REDIS_HOST", default="localhost")}:{env.int("REDIS_PORT", default=6379)}/{env("REDIS_DB", default=0)}',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'SOCKET_CONNECT_TIMEOUT': 0.5,
            'SOCKET_TIMEOUT': 0.5,
            'IGNORE_EXCEPTIONS': True,
        },
        'KEY_PREFIX': 'drf_template',
    }
}

# ================================
# 로깅 설정
# ================================

LOG_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(message)s',
            'style': '%',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'my_project': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# ================================
# i18n 및 시간 설정
# ================================

LANGUAGE_CODE = 'ko'
TIME_ZONE = 'Asia/Seoul'
USE_I18N = True
USE_TZ = True
LANGUAGES = [
    ('ko', 'Korean'),
    ('en', 'English'),
]

# ================================
# 템플릿 설정
# ================================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

USE_ETAGS = True

# ================================
# Celery 설정
# ================================

CELERY_BROKER_URL = env('CELERY_BROKER_URL', default=f'redis://{env("REDIS_HOST", default="localhost")}:{env.int("REDIS_PORT", default=6379)}/0')
CELERY_RESULT_BACKEND = env('CELERY_RESULT_BACKEND', default=f'redis://{env("REDIS_HOST", default="localhost")}:{env.int("REDIS_PORT", default=6379)}/1')

CELERY_BROKER_TRANSPORT_OPTIONS = {
    'visibility_timeout': 3600,
    'max_connections': 100,
}

CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = env.int('CELERY_TASK_TIME_LIMIT', default=1800)
CELERY_TASK_SOFT_TIME_LIMIT = env.int('CELERY_TASK_SOFT_TIME_LIMIT', default=1500)
CELERY_TIMEZONE = 'Asia/Seoul'
CELERY_ENABLE_UTC = False

# ================================
# 기본 자동 필드 설정
# ================================

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# 세션을 Redis 캐시로 저장
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [
                (
                    env('REDIS_HOST', default='localhost'),
                    env.int('REDIS_PORT', default=6379),
                )
            ],
        },
    },
}

ASGI_APPLICATION = 'Reporch.asgi.application'

# OAuth 설정 제거됨 (해커톤 템플릿에서는 필요시 추가)

# ================================
# 이메일 설정 (선택사항)
# ================================

EMAIL_BACKEND = env('EMAIL_BACKEND', default='django.core.mail.backends.console.EmailBackend')
EMAIL_HOST = env('EMAIL_HOST', default='localhost')
EMAIL_PORT = env.int('EMAIL_PORT', default=587)
EMAIL_USE_TLS = env.bool('EMAIL_USE_TLS', default=True)
EMAIL_HOST_USER = env('EMAIL_HOST_USER', default='')
EMAIL_HOST_PASSWORD = env('EMAIL_HOST_PASSWORD', default='')
DEFAULT_FROM_EMAIL = env('DEFAULT_FROM_EMAIL', default='noreply@example.com')

# ================================
# 정적 파일 경로 설정
# ================================

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# ================================
# Django Jazzmin 설정
# ================================

JAZZMIN_SETTINGS = {
    "site_title": "DRF Template Admin",
    "site_header": "DRF Template",
    "site_brand": "DRF Template",
    "site_logo": None,
    "login_logo": None,
    "login_logo_dark": None,
    "site_logo_classes": "img-circle",
    "site_icon": None,
    "welcome_sign": "DRF 템플릿 관리자 페이지에 오신 것을 환영합니다",
    "copyright": "DRF Template © 2025",
    "search_model": ["Users.CustomUser", "auth.Group"],
    "user_avatar": None,
    "topmenu_links": [
        {"name": "Home", "url": "admin:index", "permissions": ["auth.view_user"]},
        {"model": "Users.CustomUser"},
    ],
    "show_sidebar": True,
    "navigation_expanded": True,
    "hide_apps": [],
    "hide_models": [],
    "order_with_respect_to": ["auth", "Users"],
    "icons": {
        "auth": "fas fa-users-cog",
        "auth.Group": "fas fa-users",
        "Users.CustomUser": "fas fa-user-circle",
    },
    "default_icon_parents": "fas fa-chevron-circle-right",
    "default_icon_children": "fas fa-circle",
    "related_modal_active": False,
    "custom_css": None,
    "custom_js": None,
    "show_ui_builder": DEBUG,
    "changeform_format": "horizontal_tabs",
    "language_chooser": True,
}

JAZZMIN_UI_TWEAKS = {
    "navbar_small_text": False,
    "footer_small_text": False,
    "body_small_text": False,
    "brand_small_text": False,
    "brand_colour": "navbar-primary",
    "accent": "accent-primary",
    "navbar": "navbar-white navbar-light",
    "no_navbar_border": False,
    "navbar_fixed": False,
    "layout_boxed": False,
    "footer_fixed": False,
    "sidebar_fixed": False,
    "sidebar": "sidebar-dark-primary",
    "sidebar_nav_small_text": False,
    "sidebar_disable_expand": False,
    "sidebar_nav_child_indent": False,
    "sidebar_nav_compact_style": False,
    "sidebar_nav_legacy_style": False,
    "sidebar_nav_flat_style": False,
    "theme": "default",
    "dark_mode_theme": None,
    "button_classes": {
        "primary": "btn-outline-primary",
        "secondary": "btn-outline-secondary",
        "info": "btn-info",
        "warning": "btn-warning",
        "danger": "btn-danger",
        "success": "btn-success"
    }
}
