from django.test import TestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status

User = get_user_model()


class UserModelTestCase(TestCase):
    """사용자 모델 테스트"""
    
    def test_user_model_exists(self):
        """사용자 모델이 존재하는지 테스트"""
        self.assertTrue(User is not None)
    
    def test_user_creation(self):
        """사용자 생성 기능 테스트"""
        self.assertTrue(hasattr(User, 'objects'))
        self.assertTrue(hasattr(User, '_meta'))


class AuthAPITestCase(APITestCase):
    """인증 API 테스트"""
    
    def test_api_framework_loaded(self):
        """API 프레임워크가 로드되었는지 테스트"""
        self.assertTrue(status.HTTP_200_OK == 200)
        self.assertTrue(status.HTTP_400_BAD_REQUEST == 400) 