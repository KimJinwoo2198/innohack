from __future__ import annotations

from typing import Optional

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from django.core.validators import validate_email
from django.core.exceptions import ValidationError


class EmailOrUsernameModelBackend(ModelBackend):
    """
    인증 시 입력된 식별자가 이메일이면 이메일로, 아니면 사용자명으로 조회하여 로그인합니다.

    - 이메일 비교는 대소문자 구분 없이 수행합니다.
    - 사용자명 비교는 기본 동작(ModelBackend)과 동일하게 수행합니다.
    """

    def authenticate(
        self,
        request,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        if username is None or password is None:
            username = kwargs.get(get_user_model().USERNAME_FIELD)
            password = kwargs.get("password")
        if username is None or password is None:
            return None

        UserModel = get_user_model()
        is_email = self._looks_like_email(username)

        try:
            if is_email:
                user = UserModel.objects.get(email__iexact=username)
            else:
                user = UserModel.objects.get(**{UserModel.USERNAME_FIELD: username})
        except UserModel.DoesNotExist:
            UserModel().set_password(password)
            return None

        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None

    def _looks_like_email(self, value: str) -> bool:
        try:
            validate_email(value)
            return True
        except ValidationError:
            return False

