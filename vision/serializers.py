from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from rest_framework import serializers

from vision.models import (
    Food,
    FoodLog,
    FoodRating,
    FoodRecommendation,
    ResponseStyle,
)


class FoodSerializer(serializers.ModelSerializer):
    class Meta:
        model = Food
        fields = ["id", "name", "category", "nutritional_info", "default_safety_note"]


class FoodLogSerializer(serializers.ModelSerializer):
    food = FoodSerializer(read_only=True)

    class Meta:
        model = FoodLog
        fields = ["id", "meal_type", "portion", "logged_at", "notes", "food"]


class FoodRecognitionRequestSerializer(serializers.Serializer):
    image_base64 = serializers.CharField()

    def validate_image_base64(self, value: str) -> str:
        if not value.startswith("data:image") and len(value) < 100:
            raise serializers.ValidationError(_("유효한 Base64 이미지 문자열을 입력해 주세요."))
        return value


class FoodRecommendationSerializer(serializers.ModelSerializer):
    food = FoodSerializer(read_only=True)

    class Meta:
        model = FoodRecommendation
        fields = [
            "id",
            "food",
            "priority",
            "is_safe",
            "safety_info",
            "nutritional_advice",
            "reasoning",
            "created_at",
        ]


class FoodRatingSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = FoodRating
        fields = ["id", "user", "food", "rating", "comment", "created_at", "updated_at"]
        read_only_fields = ["created_at", "updated_at"]

    def validate_rating(self, value: int) -> int:
        if value < 1 or value > 5:
            raise serializers.ValidationError(_("평점은 1~5 사이여야 합니다."))
        return value

    def create(self, validated_data):
        user = validated_data["user"]
        food = validated_data["food"]
        rating, _ = FoodRating.objects.update_or_create(
            user=user,
            food=food,
            defaults={"rating": validated_data["rating"], "comment": validated_data.get("comment", "")},
        )
        return rating


class FoodRatingSummarySerializer(serializers.Serializer):
    food_id = serializers.IntegerField()
    average_rating = serializers.FloatField()
    total_ratings = serializers.IntegerField()
    high_ratio = serializers.FloatField()
    low_ratio = serializers.FloatField()


class ResponseStyleSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResponseStyle
        fields = ["id", "name", "prompt", "description", "is_default", "created_at"]


class ResponseStylePreferenceSerializer(serializers.Serializer):
    response_style_id = serializers.PrimaryKeyRelatedField(
        queryset=ResponseStyle.objects.all(), source="response_style"
    )

