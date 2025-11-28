from __future__ import annotations

import logging
from typing import Dict, List

from django.db import transaction

from vision.models import Food, FoodRecommendation
from vision.services.nutrition_analysis import analyze_nutrients
from vision.utils.cache import build_cache_key, get_cache_value, set_cache_value

logger = logging.getLogger(__name__)

RECOMMENDATION_CACHE_TTL = 3600


def _find_deficient_nutrients(analysis: Dict[str, List[Dict[str, object]]], threshold: float) -> List[str]:
    deficient = []
    for nutrient in analysis.get("nutrients", []):
        percentage = nutrient.get("percentage")
        if percentage is None or percentage >= threshold:
            continue
        deficient.append(nutrient["nutrient"])
    return deficient


def generate_personalized_recommendations(user, threshold: float = 70.0, limit: int = 5) -> Dict[str, object]:
    analysis = analyze_nutrients(user)
    cache_suffix = f"{analysis.get('start')}:{analysis.get('end')}"
    cache_key = build_cache_key(user.id, "recommendations", cache_suffix)
    cached = get_cache_value(cache_key)
    if cached:
        return cached

    deficient = _find_deficient_nutrients(analysis, threshold)
    if not deficient:
        payload = {"deficient_nutrients": [], "recommendation_ids": []}
        set_cache_value(cache_key, payload, RECOMMENDATION_CACHE_TTL)
        return payload

    foods = (
        Food.objects.filter(is_active=True, nutritional_info__has_any_keys=deficient)
        .order_by("name")[:limit]
    )

    recommendation_ids: List[int] = []
    with transaction.atomic():
        for idx, food in enumerate(foods, start=1):
            priority = min(idx, 10)
            recommendation, _ = FoodRecommendation.objects.update_or_create(
                user=user,
                food=food,
                defaults={
                    "priority": priority,
                    "reasoning": f"{', '.join(deficient)} 영양소 보충용 추천",
                    "is_safe": True,
                    "safety_info": food.default_safety_note,
                    "nutritional_advice": "부족한 영양소를 보완하도록 소량부터 섭취하세요.",
                },
            )
            recommendation_ids.append(recommendation.id)

    payload = {"deficient_nutrients": deficient, "recommendation_ids": recommendation_ids}
    set_cache_value(cache_key, payload, RECOMMENDATION_CACHE_TTL)
    return payload


