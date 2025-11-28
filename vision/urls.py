from django.urls import include, path
from rest_framework.routers import DefaultRouter

from vision.views import (
    FoodRatingViewSet,
    FoodRecognitionAPIView,
    NutrientAnalysisView,
    PersonalizedRecommendationView,
    ResponseStylePreferenceView,
    ResponseStyleViewSet,
)

router = DefaultRouter()
router.register(r"food-ratings", FoodRatingViewSet, basename="food-rating")
router.register(r"response-styles", ResponseStyleViewSet, basename="response-style")

urlpatterns = [
    path("foods/recognize/", FoodRecognitionAPIView.as_view(), name="vision-food-recognize"),
    path("food-logs/nutrient-analysis/", NutrientAnalysisView.as_view(), name="vision-nutrient-analysis"),
    path("food-recommendations/personalized/", PersonalizedRecommendationView.as_view(), name="vision-personalized-recommendations"),
    path("response-styles/preference/", ResponseStylePreferenceView.as_view(), name="vision-response-style-preference"),
    path("", include(router.urls)),
]


