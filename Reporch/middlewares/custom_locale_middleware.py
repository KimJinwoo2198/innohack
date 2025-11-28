from django.utils import translation
from django.utils.deprecation import MiddlewareMixin

class CustomLocaleMiddleware(MiddlewareMixin):
    def process_request(self, request):
        # Admin 이외는 모두 API로 간주: DB 조회 없이 Accept-Language 기반으로만 적용
        path = getattr(request, 'path', '')
        if not path.startswith('/admin/'):
            language = translation.get_language_from_request(request)
            translation.activate(language)
            request.LANGUAGE_CODE = translation.get_language()
            return None

        # Admin만 사용자 선호 언어 고려 (세션 기반)
        language = None
        if request.COOKIES.get('sessionid'):
            user = getattr(request, 'user', None)
            is_authenticated = bool(getattr(user, 'is_authenticated', False))
            if is_authenticated:
                pref = getattr(user, 'language_preference', None)
                if pref:
                    language = pref

        if not language:
            language = translation.get_language_from_request(request)

        translation.activate(language)
        request.LANGUAGE_CODE = translation.get_language()
        return None

    def process_response(self, _request, response):
        # 캐시가 언어별로 분리되도록 헤더 설정
        lang = translation.get_language()
        if lang:
            response['Content-Language'] = lang
            # Accept-Language 기반 캐시 분리
            vary = response.get('Vary')
            if vary:
                if 'Accept-Language' not in vary:
                    response['Vary'] = f"{vary}, Accept-Language"
            else:
                response['Vary'] = 'Accept-Language'
        return response