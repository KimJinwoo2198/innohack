from rest_framework.exceptions import APIException
from rest_framework import status

class InvalidCredentialsException(APIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    default_detail = '아이디 또는 비밀번호가 잘못되었습니다.'
    default_code = 'invalid_credentials'

class AccountInactiveException(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_detail = '계정이 비활성화되었습니다. 이메일을 확인해 주세요.'
    default_code = 'account_inactive'

class UserCreationError(Exception):
    """Exception raised for errors during user creation."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)