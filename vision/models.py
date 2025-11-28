from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


class TimeStampedModel(models.Model):
    """공통 생성/수정 타임스탬프 모델."""

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class PregnancyStage(TimeStampedModel):
    """임신 주차별 스테이지 정의."""

    name = models.CharField(max_length=128)
    start_week = models.PositiveIntegerField()
    end_week = models.PositiveIntegerField()
    description = models.TextField(blank=True)

    class Meta:
        ordering = ["start_week"]
        indexes = [
            models.Index(fields=["start_week", "end_week"]),
        ]
        unique_together = ("start_week", "end_week")

    def __str__(self) -> str:
        return f"{self.name} ({self.start_week}-{self.end_week}주)"


class NutrientRequirement(TimeStampedModel):
    """주차별 영양 요구량."""

    stage = models.ForeignKey(
        PregnancyStage, on_delete=models.CASCADE, related_name="nutrient_requirements"
    )
    nutrient_name = models.CharField(max_length=128)
    daily_value = models.DecimalField(max_digits=10, decimal_places=2)
    unit = models.CharField(max_length=32, default="mg")
    description = models.TextField(blank=True)

    class Meta:
        ordering = ["nutrient_name"]
        unique_together = ("stage", "nutrient_name")

    def __str__(self) -> str:
        return f"{self.stage.name} - {self.nutrient_name}"


class Food(TimeStampedModel):
    """음식 데이터 및 영양 성분."""

    name = models.CharField(max_length=255, unique=True)
    category = models.CharField(max_length=128, blank=True)
    nutritional_info = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    default_safety_note = models.TextField(blank=True)

    class Meta:
        ordering = ["name"]
        indexes = [
            models.Index(fields=["name"]),
        ]

    def __str__(self) -> str:
        return str(self.name)


class UserPregnancyProfile(TimeStampedModel):
    """사용자 임신 프로필 및 캐시된 스테이지 정보."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="pregnancy_profile"
    )
    current_week = models.PositiveIntegerField(default=1)
    pre_pregnancy_bmi = models.DecimalField(
        max_digits=5, decimal_places=2, blank=True, null=True
    )
    weight_gain_kg = models.DecimalField(
        max_digits=5, decimal_places=2, blank=True, null=True
    )
    stage_cached_id = models.BigIntegerField(blank=True, null=True)
    stage_cached_label = models.CharField(max_length=255, blank=True)
    stage_cached_until = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ["-updated_at"]

    def cache_stage(self, stage: PregnancyStage, ttl_hours: int = 24) -> None:
        self.stage_cached_id = stage.id
        self.stage_cached_label = stage.name
        self.stage_cached_until = timezone.now() + timedelta(hours=ttl_hours)
        self.save(update_fields=["stage_cached_id", "stage_cached_label", "stage_cached_until"])

    def is_stage_cache_valid(self) -> bool:
        return bool(self.stage_cached_until and self.stage_cached_until > timezone.now())

    def __str__(self) -> str:
        user_display: str = str(self.user)
        return f"{user_display} - {self.current_week}주"


class FoodLog(TimeStampedModel):
    """사용자 음식 섭취 기록."""

    class MealType(models.TextChoices):
        BREAKFAST = "breakfast", "아침"
        LUNCH = "lunch", "점심"
        DINNER = "dinner", "저녁"
        SNACK = "snack", "간식"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="food_logs"
    )
    food = models.ForeignKey(Food, on_delete=models.CASCADE, related_name="food_logs")
    meal_type = models.CharField(max_length=16, choices=MealType.choices)
    portion = models.DecimalField(max_digits=6, decimal_places=2, default=1)
    logged_at = models.DateTimeField(default=timezone.now, db_index=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-logged_at"]
        indexes = [
            models.Index(fields=["user", "logged_at"]),
        ]

    def __str__(self) -> str:
        user_display = str(self.user)
        return f"{user_display} - {self.food.name}"


class ResponseStyle(TimeStampedModel):
    """AI 응답 화법 정의."""

    name = models.CharField(max_length=64, unique=True)
    prompt = models.TextField()
    description = models.CharField(max_length=255, blank=True)
    is_default = models.BooleanField(default=False)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return str(self.name)


class FoodRecommendation(TimeStampedModel):
    """AI 개인화 음식 추천."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="food_recommendations"
    )
    food = models.ForeignKey(Food, on_delete=models.CASCADE, related_name="recommendations")
    priority = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)]
    )
    is_safe = models.BooleanField(default=True)
    safety_info = models.TextField(blank=True)
    nutritional_advice = models.TextField(blank=True)
    reasoning = models.TextField(blank=True)

    class Meta:
        ordering = ["priority", "-created_at"]
        unique_together = ("user", "food")

    def __str__(self) -> str:
        user_display: str = str(self.user)
        return f"{user_display} -> {self.food.name} ({self.priority})"


class FoodRating(TimeStampedModel):
    """사용자 음식 평가."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="food_ratings"
    )
    food = models.ForeignKey(Food, on_delete=models.CASCADE, related_name="ratings")
    rating = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    comment = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        unique_together = ("user", "food")

    def __str__(self) -> str:
        return f"{self.food.name} - {self.rating}"


class FoodRecognitionLog(TimeStampedModel):
    """비전 인식 호출 로그."""

    class Status(models.TextChoices):
        SUCCESS = "success", "성공"
        FAILURE = "failure", "실패"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="vision_logs"
    )
    image_sha = models.CharField(max_length=128, db_index=True)
    food_name = models.CharField(max_length=255, blank=True)
    confidence_score = models.DecimalField(
        max_digits=4,
        decimal_places=3,
        default=Decimal("0.0"),
        validators=[MinValueValidator(0), MaxValueValidator(1)],
    )
    raw_response = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.SUCCESS)
    error_message = models.TextField(blank=True)
    image_placeholder = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["image_sha"]),
        ]

    def __str__(self) -> str:
        user_display = str(self.user)
        return f"{user_display} - {self.status}"


