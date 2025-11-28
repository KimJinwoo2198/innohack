import logging

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated

from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone

from .serializers import (
    CustomUserCreationSerializer,
    CustomAuthTokenSerializer,
    EmptySerializer,
)
from .tasks import warm_profile_cache

from drf_spectacular.utils import (
    extend_schema,
)

logger = logging.getLogger(__name__)
User = get_user_model()

class SignupViewSet(viewsets.GenericViewSet):
    permission_classes = [AllowAny]
    serializer_class = CustomUserCreationSerializer

    @extend_schema(
        summary="회원가입",
        description="새로운 사용자를 등록합니다.",
        request=CustomUserCreationSerializer,
        tags=["Authentication"]
    )
    @action(detail=False, methods=['post'], url_path='signup')
    def create_account(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            try:
                user = serializer.save()
                user.is_active = True
                user.save()
                
                return Response({
                    "status": "success",
                    "code": 201,
                    "message": "회원가입이 완료되었습니다.",
                    "data": {"user_id": user.id},
                    "meta": {"timestamp": timezone.now().isoformat()}
                }, status=status.HTTP_201_CREATED)
            except Exception as e:
                logger.error("회원가입 중 오류 발생", exc_info=True)
                return Response({
                    "status": "error",
                    "code": 500,
                    "message": "회원가입 처리 중 문제가 발생했습니다. 다시 시도해주세요.",
                    "error_code": "signup_error",
                    "meta": {"timestamp": timezone.now().isoformat()}
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({
                "status": "error",
                "code": 400,
                "message": "입력 데이터가 유효하지 않습니다.",
                "errors": serializer.errors,
                "meta": {"timestamp": timezone.now().isoformat()}
            }, status=status.HTTP_400_BAD_REQUEST)

class LoginViewSet(viewsets.GenericViewSet):
    permission_classes = [AllowAny]
    serializer_class = CustomAuthTokenSerializer

    @extend_schema(
        summary="로그인",
        description="사용자 로그인 및 JWT 토큰 발급",
        request=CustomAuthTokenSerializer,
        tags=["Authentication"]
    )
    @action(detail=False, methods=['post'], authentication_classes=[])
    def login(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data, context={'request': request})
        try:
            serializer.is_valid(raise_exception=True)
            user = serializer.validated_data['user']

            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)
            response = Response({
                "status": "success",
                "code": status.HTTP_200_OK,
                "message": "로그인에 성공했습니다."
            }, status=status.HTTP_200_OK)
            warm_profile_cache.delay(user.id)
            response.set_cookie(
                key=settings.JWT_ACCESS_COOKIE_NAME,
                value=access_token,
                httponly=settings.JWT_COOKIE_HTTPONLY,
                secure=settings.JWT_COOKIE_SECURE,
                samesite=settings.JWT_COOKIE_SAMESITE,
                domain=settings.JWT_COOKIE_DOMAIN,
                path=settings.JWT_COOKIE_PATH,
            )
            response.set_cookie(
                key=settings.JWT_REFRESH_COOKIE_NAME,
                value=str(refresh),
                httponly=settings.JWT_COOKIE_HTTPONLY,
                secure=settings.JWT_COOKIE_SECURE,
                samesite=settings.JWT_COOKIE_SAMESITE,
                domain=settings.JWT_COOKIE_DOMAIN,
                path=settings.JWT_COOKIE_PATH,
            )
            return response
        except Exception:
            raise

class LogoutViewSet(viewsets.GenericViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = EmptySerializer

    @extend_schema(
        summary="로그아웃",
        description="JWT 토큰을 블랙리스트에 추가하고 쿠키를 제거합니다.",
        tags=["Authentication"]
    )
    @action(detail=False, methods=['post'], url_path='logout')
    def logout(self, request):
        refresh_token = request.COOKIES.get(settings.JWT_REFRESH_COOKIE_NAME)
        if not refresh_token:
            return Response({"status":"error","code":400,"message":"리프레시 토큰이 없습니다."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            token = RefreshToken(refresh_token)
            token.blacklist()
        except Exception as e:
            logger.error("Logout error", exc_info=True)
            return Response({
                "status": "error",
                "code": 500,
                "message": "로그아웃 처리 중 문제가 발생했습니다.",
                "error_code": "logout_error",
                "meta": {"timestamp": timezone.now().isoformat()}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        response = Response({"status":"success","code":200,"message":"로그아웃 되었습니다."}, status=status.HTTP_200_OK)
        response.delete_cookie(
            settings.JWT_ACCESS_COOKIE_NAME,
            path=settings.JWT_COOKIE_PATH,
            domain=settings.JWT_COOKIE_DOMAIN,
            samesite=settings.JWT_COOKIE_SAMESITE
        )
        response.delete_cookie(
            settings.JWT_REFRESH_COOKIE_NAME,
            path=settings.JWT_COOKIE_PATH,
            domain=settings.JWT_COOKIE_DOMAIN,
            samesite=settings.JWT_COOKIE_SAMESITE
        )
        return response