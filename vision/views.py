from __future__ import annotations

import logging
from decimal import Decimal

from django.db.models import Avg, Count, Q
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from vision.models import FoodRating, FoodRecognitionLog, FoodRecommendation, ResponseStyle
from vision.serializers import (
    FoodRatingSerializer,
    FoodRatingSummarySerializer,
    FoodRecognitionRequestSerializer,
    FoodRecommendationSerializer,
    ResponseStylePreferenceSerializer,
    ResponseStyleSerializer,
)
from vision.services import (
    get_food_guidance,
    generate_personalized_recommendations,
    recognize_food_from_image,
)
from vision.services.nutrition_analysis import analyze_nutrients
from vision.utils.context import get_response_style_prompt

logger = logging.getLogger(__name__)


def error_response(code: str, message: str, status_code: int) -> Response:
    return Response({"error": {"code": code, "message": message}}, status=status_code)


class FoodRecognitionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FoodRecognitionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        style_prompt = get_response_style_prompt(request.user)
        try:
            payload, image_sha, cache_hit = recognize_food_from_image(
                request.user, serializer.validated_data["image_base64"], style_prompt
            )
        except Exception as exc:  # pragma: no cover - OpenAI runtime issues
            logger.exception("음식 인식 실패 user=%s error=%s", request.user.id, exc)
            FoodRecognitionLog.objects.create(
                user=request.user,
                image_sha="",
                image_placeholder="",
                status=FoodRecognitionLog.Status.FAILURE,
                error_message=str(exc),
            )
            return error_response("vision_error", "음식 인식에 실패했습니다.", status.HTTP_500_INTERNAL_SERVER_ERROR)

        food_name = payload.get("food_name")
        if not food_name:
            return error_response("vision_invalid_response", "음식명을 찾을 수 없습니다.", status.HTTP_400_BAD_REQUEST)

        if food_name.lower() == "unknown":
            FoodRecognitionLog.objects.create(
                user=request.user,
                image_sha=image_sha,
                image_placeholder=f"sha256://{image_sha[:16]}",
                status=FoodRecognitionLog.Status.FAILURE,
                raw_response=payload,
                error_message="Unknown food",
            )
            return error_response("vision_food_unknown", "음식을 식별하지 못했습니다.", status.HTTP_404_NOT_FOUND)

        guidance = get_food_guidance(request.user, food_name, style_prompt)
        confidence = payload.get("confidence", 0.8)
        try:
            confidence_decimal = Decimal(str(confidence))
        except Exception:
            confidence_decimal = Decimal("0.8")

        log = FoodRecognitionLog.objects.create(
            user=request.user,
            image_sha=image_sha,
            image_placeholder=f"sha256://{image_sha[:16]}",
            food_name=food_name,
            confidence_score=confidence_decimal,
            raw_response=payload,
            status=FoodRecognitionLog.Status.SUCCESS,
        )

        response_payload = {
            "food_name": food_name,
            "confidence": float(confidence_decimal),
            "is_safe": guidance.get("is_safe", False),
            "safety_info": guidance.get("safety_summary"),
            "nutritional_advice": guidance.get("nutritional_advice"),
            "model": payload.get("model"),
            "cache_hit": cache_hit,
            "log_id": log.id,
        }
        return Response(response_payload, status=status.HTTP_200_OK)


class NutrientAnalysisView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        analysis = analyze_nutrients(request.user)
        return Response(analysis, status=status.HTTP_200_OK)


class PersonalizedRecommendationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        result = generate_personalized_recommendations(request.user)
        recommendations = FoodRecommendation.objects.filter(id__in=result["recommendation_ids"]).select_related("food")
        serialized = FoodRecommendationSerializer(recommendations, many=True).data
        payload = {
            "deficient_nutrients": result["deficient_nutrients"],
            "recommendations": serialized,
        }
        return Response(payload, status=status.HTTP_200_OK)


class FoodRatingViewSet(viewsets.ModelViewSet):
    serializer_class = FoodRatingSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return FoodRating.objects.filter(user=self.request.user).select_related("food")

    @action(detail=False, methods=["get"])
    def summary(self, request):
        food_id = request.query_params.get("food_id")
        if not food_id:
            return error_response("missing_food_id", "food_id가 필요합니다.", status.HTTP_400_BAD_REQUEST)
        try:
            summary = (
                FoodRating.objects.filter(food_id=food_id)
                .aggregate(
                    avg=Avg("rating"),
                    total=Count("id"),
                    high=Count("id", filter=Q(rating__gte=4)),
                    low=Count("id", filter=Q(rating__lte=2)),
                )
            )
        except Exception as exc:
            logger.exception("평점 요약 실패 food_id=%s error=%s", food_id, exc)
            return error_response("rating_summary_failed", "평점 요약에 실패했습니다.", status.HTTP_500_INTERNAL_SERVER_ERROR)

        total = summary["total"] or 0
        high_ratio = (summary["high"] / total) if total else 0.0
        low_ratio = (summary["low"] / total) if total else 0.0
        payload = FoodRatingSummarySerializer(
            {
                "food_id": int(food_id),
                "average_rating": summary["avg"] or 0.0,
                "total_ratings": total,
                "high_ratio": round(high_ratio, 2),
                "low_ratio": round(low_ratio, 2),
            }
        ).data
        return Response(payload, status=status.HTTP_200_OK)


class ResponseStyleViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin, mixins.UpdateModelMixin, viewsets.GenericViewSet):
    queryset = ResponseStyle.objects.all().order_by("name")
    serializer_class = ResponseStyleSerializer
    permission_classes = [IsAuthenticated]

    def get_permissions(self):
        if self.action in {"create", "update", "partial_update"}:
            return [IsAdminUser()]
        return super().get_permissions()


class ResponseStylePreferenceView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ResponseStylePreferenceSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        response_style = serializer.validated_data["response_style"]
        request.user.preferred_speaking_style = response_style
        request.user.save(update_fields=["preferred_speaking_style"])
        return Response(ResponseStyleSerializer(response_style).data, status=status.HTTP_200_OK)

