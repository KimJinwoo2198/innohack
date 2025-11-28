from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import RegexValidator
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.hashers import make_password, check_password
from utils.Snowflake import SnowflakeGenerator
from django.utils import timezone
import phonenumbers  # type: ignore  # pylint: disable=import-error
from django.core.exceptions import ValidationError


snowflake_generator = SnowflakeGenerator(worker_id=1,epoch=1609459200000)

def generate_unique_id():
    return snowflake_generator.get_id(12)

USER_ROLES = (
    ('admin', '관리자'),
    ('user', '일반 사용자'),
)

class CustomUser(AbstractUser):
    id = models.BigIntegerField(
        primary_key=True,
        unique=True,
        editable=False,
        default=generate_unique_id 
    )
    email = models.EmailField(_('email address'), unique=True, db_index=True)
    phone_number = models.CharField(
        max_length=16,
        unique=True,
        blank=True,
        null=True,
        db_index=True
    )
    profile_image = models.ImageField(
        upload_to='avatars/',
        default='avatars/defaultprofile.png',
        blank=True
    )
    role = models.CharField(
        max_length=20,
        choices=USER_ROLES,
        default='user'
    )
    score = models.IntegerField(default=0)
    social_account_type = models.CharField(
        max_length=20,
        choices=(('google', 'Google'), ('github', 'GitHub')),
        null=True,
        blank=True
    )
    social_id = models.CharField(max_length=100, null=True, blank=True)
    display_name = models.CharField(max_length=50, blank=True, null=True)
    language_preference = models.CharField(
        max_length=10,
        choices=[('en', 'English'), ('ko', 'Korean')],
        default='ko'
    )
    has_active_subscription = models.BooleanField(default=False, db_index=True)
    preferred_speaking_style = models.ForeignKey(
        'vision.ResponseStyle',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='preferred_users',
        help_text=_('사용자가 선호하는 응답 화법'),
    )

    def save(self, *args, **kwargs):
        if self.phone_number:
            try:
                phone_obj = phonenumbers.parse(self.phone_number, 'KR')
                if not phonenumbers.is_valid_number(phone_obj):
                    raise ValidationError({'phone_number': _('유효한 전화번호가 아닙니다.')})
                self.phone_number = phonenumbers.format_number(
                    phone_obj, phonenumbers.PhoneNumberFormat.E164
                )
            except phonenumbers.NumberParseException:
                raise ValidationError({'phone_number': _('유효한 전화번호가 아닙니다.')})
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return str(self.username)

    class Meta:
        indexes = [
            models.Index(fields=["id"]),
            models.Index(fields=["email"]),
        ]
