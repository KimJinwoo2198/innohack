from rest_framework.views import exception_handler
from django.http import JsonResponse
from django.utils.timezone import now
from django.db import DatabaseError, IntegrityError, OperationalError
from django.core.exceptions import PermissionDenied as DjangoPermissionDenied, ValidationError as DjangoValidationError
from rest_framework.exceptions import AuthenticationFailed, NotAuthenticated, PermissionDenied as DRFPermissionDenied, ValidationError as DRFValidationError
from .exceptions import (
    InvalidCredentialsException,
    AccountInactiveException,
)

def custom_exception_handler(exc, context):
    # 먼저 DRF 기본 예외 핸들러를 호출합니다
    response = exception_handler(exc, context)

    # 인증/권한 관련 예외 처리
    if isinstance(exc, (AuthenticationFailed, NotAuthenticated)):
        return JsonResponse({
            "status": "error",
            "code": 401,
            "message": "인증이 필요합니다.",
            "error_code": "authentication_required",
            "meta": {"timestamp": now().isoformat()}
        }, status=401)
    if isinstance(exc, (DRFPermissionDenied, DjangoPermissionDenied)):
        return JsonResponse({
            "status": "error",
            "code": 403,
            "message": "권한이 없습니다.",
            "error_code": "permission_denied",
            "meta": {"timestamp": now().isoformat()}
        }, status=403)

    # 유효성 검사 오류 처리
    if isinstance(exc, (DRFValidationError, DjangoValidationError)):
        detail = exc.detail if hasattr(exc, 'detail') else None
        return JsonResponse({
            "status": "error",
            "code": 400,
            "message": "유효성 검사 오류가 발생했습니다.",
            "error_code": "validation_error",
            "errors": detail,
            "meta": {"timestamp": now().isoformat()}
        }, status=400)

    # 데이터베이스 관련 예외 처리
    if isinstance(exc, IntegrityError):
        return JsonResponse({
            "status": "error",
            "code": 409,
            "message": "데이터 무결성 오류가 발생했습니다.",
            "error_code": "integrity_error",
            "meta": {"timestamp": now().isoformat()}
        }, status=409)
    if isinstance(exc, OperationalError):
        return JsonResponse({
            "status": "error",
            "code": 500,
            "message": "데이터베이스 연결 오류가 발생했습니다.",
            "error_code": "database_connection_error",
            "meta": {"timestamp": now().isoformat()}
        }, status=500)
    if isinstance(exc, DatabaseError):
        return JsonResponse({
            "status": "error",
            "code": 500,
            "message": "데이터베이스 오류가 발생했습니다.",
            "error_code": "database_error",
            "meta": {"timestamp": now().isoformat()}
        }, status=500)

    if isinstance(exc, (InvalidCredentialsException, AccountInactiveException)):
        return JsonResponse({
            "status": "error",
            "code": exc.status_code,
            "message": exc.default_detail,
            "error_code": exc.default_code,
            "meta": {
                "timestamp": now().isoformat()
            }
        }, status=exc.status_code)

    if response is not None:
        # DRF 기본 응답 포맷으로 변환
        response.data = {
            "status": "error",
            "code": response.status_code,
            "message": response.data.get('detail', '요청 처리 중 오류가 발생했습니다.'),
            "meta": {
                "timestamp": now().isoformat()
            }
        }

    return response
