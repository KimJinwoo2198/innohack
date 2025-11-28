from django.utils.deprecation import MiddlewareMixin


class APIMinimalMiddleware(MiddlewareMixin):
    """
    /api/ 경로에서 세션 및 CSRF 관련 쿠키를 제거하여
    SessionMiddleware/AuthenticationMiddleware의 불필요한 비용을 피한다.
    """

    def process_request(self, request):
        path = getattr(request, 'path', '')
        # Admin 외의 모든 경로를 API로 간주하여 최소 오버헤드로 처리
        if path.startswith('/admin/'):
            return None
        cookies = request.COOKIES
        # 세션 쿠키 제거 → 세션 로드/유저 로딩 방지
        if 'sessionid' in cookies:
            # 키 존재 확인 후 삭제만 수행 (예외 발생 가능성 낮음)
            del cookies['sessionid']
        # CSRF 쿠키 제거 (DRF APIView는 기본적으로 CSRF exempt)
        if 'csrftoken' in cookies:
            del cookies['csrftoken']
        return None

