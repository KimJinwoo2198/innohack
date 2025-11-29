from __future__ import annotations

import logging
from typing import Dict, Optional

from django.db import transaction

from vision.models import PregnancyStage, ResponseStyle, UserPregnancyProfile

logger = logging.getLogger(__name__)

DEFAULT_WEEK_CONTEXT = {
    "current_week": "20",
    "stage_label": "미정",
    "weight_gain_kg": "",
}


def get_or_create_profile(user) -> UserPregnancyProfile:
    profile, _ = UserPregnancyProfile.objects.get_or_create(user=user)
    return profile


def resolve_pregnancy_stage(profile: UserPregnancyProfile) -> Optional[PregnancyStage]:
    if profile.is_stage_cache_valid() and profile.stage_cached_id:
        try:
            return PregnancyStage.objects.get(id=profile.stage_cached_id)
        except PregnancyStage.DoesNotExist:
            logger.warning("Cached stage id %s no longer exists", profile.stage_cached_id)

    stage = (
        PregnancyStage.objects.filter(
            start_week__lte=profile.current_week, end_week__gte=profile.current_week
        )
        .order_by("start_week")
        .first()
    )
    if stage:
        profile.cache_stage(stage)
    return stage


def build_week_context(user) -> Dict[str, str]:
    if not user or not getattr(user, "is_authenticated", False):
        return DEFAULT_WEEK_CONTEXT.copy()
    profile = get_or_create_profile(user)
    stage = resolve_pregnancy_stage(profile)
    return {
        "current_week": str(profile.current_week),
        "stage_label": stage.name if stage else "미정",
        "weight_gain_kg": str(profile.weight_gain_kg or ""),
    }


def get_response_style(user) -> Optional[ResponseStyle]:
    style = getattr(user, "preferred_speaking_style", None)
    if style:
        return style
    return ResponseStyle.objects.filter(is_default=True).order_by("name").first()


def ensure_default_response_style() -> ResponseStyle | None:
    style = ResponseStyle.objects.filter(is_default=True).first()
    if style:
        return style
    with transaction.atomic():
        style = ResponseStyle.objects.filter(is_default=True).first()
        if style:
            return style
        return ResponseStyle.objects.create(
            name="default",
            prompt="따뜻하고 존중하는 한국어 화법으로 산모 안전과 영양 정보를 제공합니다.",
            description="기본 화법",
            is_default=True,
        )


def get_response_style_prompt(user) -> str:
    style = get_response_style(user) or ensure_default_response_style()
    if style:
        return style.prompt
    return "따뜻하고 존중하는 한국어 화법으로 안전 정보를 제공합니다."


