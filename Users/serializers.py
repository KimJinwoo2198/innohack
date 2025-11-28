from rest_framework import serializers
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from .models import CustomUser
from .exceptions import InvalidCredentialsException, AccountInactiveException
from django.utils.translation import gettext_lazy as _
from django.core.validators import RegexValidator
import re
import logging
import phonenumbers

logger = logging.getLogger(__name__)

class CustomUserCreationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
        help_text='비밀번호는 최소 8자 이상이어야 하며, 숫자와 특수문자를 포함해야 합니다.'
    )
    password2 = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
        help_text='비밀번호 확인을 위해 다시 입력해주세요.'
    )
    email = serializers.EmailField(
        required=True,
        help_text='유효한 이메일 주소를 입력해주세요.'
    )
    phone_number = serializers.CharField(
        required=True,
        help_text='유효한 전화번호를 입력해주세요. 예: 010-1234-5678'
    )

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password', 'password2', 'phone_number']

    def validate_username(self, value):
        if CustomUser.objects.filter(username=value).exists():
            raise serializers.ValidationError({"username": "이미 사용 중인 사용자 이름입니다."})
        if not re.match(r'^[\w.@+-]+$', value):
            raise serializers.ValidationError({"username": "사용자 이름은 문자, 숫자 및 @/./+/-/_만 포함할 수 있습니다."})
        return value

    def validate_email(self, value):
        if CustomUser.objects.filter(email=value).exists():
            raise serializers.ValidationError({"email": "이미 등록된 이메일 주소입니다."})
        return value

    def validate_phone_number(self, value):
        try:
            phone_obj = phonenumbers.parse(value, "KR")
            if not phonenumbers.is_valid_number(phone_obj):
                raise serializers.ValidationError({"phone_number": _("유효한 전화번호가 아닙니다.")})
            normalized = phonenumbers.format_number(
                phone_obj, phonenumbers.PhoneNumberFormat.E164
            )
        except phonenumbers.NumberParseException:
            raise serializers.ValidationError({"phone_number": _("유효한 전화번호가 아닙니다.")})
        if CustomUser.objects.filter(phone_number=normalized).exists():
            raise serializers.ValidationError({"phone_number": _("이미 등록된 전화번호입니다.")})
        return normalized

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "비밀번호가 일치하지 않습니다."})

        try:
            validate_password(attrs['password'])
        except ValidationError as e:
            raise serializers.ValidationError({"password": list(e.messages)})

        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        user = CustomUser.objects.create_user(**validated_data)
        user.is_active = True
        user.save()
        return user

class CustomAuthTokenSerializer(serializers.Serializer):
    username = serializers.CharField(label=_("Username"), write_only=True)
    password = serializers.CharField(label=_("Password"), style={'input_type': 'password'}, trim_whitespace=False, write_only=True)

    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')

        if username and password:
            user = authenticate(request=self.context.get('request'), username=username, password=password)

            if not user:
                raise InvalidCredentialsException()
            if not user.is_active:
                raise AccountInactiveException()
        else:
            raise serializers.ValidationError({
                "detail": _('Must include "username" and "password".')
            }, code='authorization')

        attrs['user'] = user
        return attrs

class EmptySerializer(serializers.Serializer):
    pass