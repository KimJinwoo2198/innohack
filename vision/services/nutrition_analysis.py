from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal
from typing import Dict, List

from django.utils import timezone

from vision.models import FoodLog, NutrientRequirement, PregnancyStage, UserPregnancyProfile
from vision.utils.cache import build_cache_key, get_cache_value, set_cache_value

logger = logging.getLogger(__name__)

ANALYSIS_CACHE_TTL = 3600


def _get_stage_for_user(user) -> PregnancyStage | None:
    profile, _ = UserPregnancyProfile.objects.get_or_create(user=user)
    return (
        PregnancyStage.objects.filter(
            start_week__lte=profile.current_week, end_week__gte=profile.current_week
        )
        .order_by("start_week")
        .first()
    )


def _fetch_requirements(stage: PregnancyStage | None) -> Dict[str, Decimal]:
    if not stage:
        return {}
    requirements = NutrientRequirement.objects.filter(stage=stage)
    return {req.nutrient_name: req.daily_value for req in requirements}


def _aggregate_nutrients(logs) -> Dict[str, Decimal]:
    totals: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for log in logs:
        nutritional_info = log.food.nutritional_info or {}
        for nutrient, value in nutritional_info.items():
            try:
                amount = Decimal(str(value)) * Decimal(str(log.portion))
            except Exception:
                logger.warning("영양 정보 변환 실패 nutrient=%s value=%s", nutrient, value)
                continue
            totals[nutrient] += amount
    return totals


def analyze_nutrients(user, days: int = 7) -> Dict[str, List[Dict[str, object]]]:
    end = timezone.now()
    start = end - timedelta(days=days)
    cache_suffix = f"{start.date()}:{end.date()}"
    cache_key = build_cache_key(user.id, "nutrient_analysis", cache_suffix)
    cached = get_cache_value(cache_key)
    if cached:
        return cached

    logs = (
        FoodLog.objects.select_related("food")
        .filter(user=user, logged_at__gte=start)
        .order_by("-logged_at")
    )
    stage = _get_stage_for_user(user)
    requirements = _fetch_requirements(stage)
    totals = _aggregate_nutrients(logs)

    nutrients_report = []
    handled = set()
    for nutrient, consumed in totals.items():
        required = requirements.get(nutrient)
        percentage = None
        if required and required > 0:
            percentage = float((consumed / required) * Decimal("100"))
        nutrients_report.append(
            {
                "nutrient": nutrient,
                "consumed": float(consumed),
                "required": float(required) if required else None,
                "percentage": round(percentage, 2) if percentage is not None else None,
            }
        )
        handled.add(nutrient)

    for nutrient, required in requirements.items():
        if nutrient in handled:
            continue
        percentage = 0.0 if required else None
        nutrients_report.append(
            {
                "nutrient": nutrient,
                "consumed": 0.0,
                "required": float(required) if required else None,
                "percentage": percentage,
            }
        )

    payload = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "stage": stage.name if stage else None,
        "nutrients": nutrients_report,
    }
    set_cache_value(cache_key, payload, ANALYSIS_CACHE_TTL)
    return payload

