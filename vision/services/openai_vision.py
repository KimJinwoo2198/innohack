from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from typing import Any, Dict, Tuple

from django.conf import settings
from openai import OpenAI, OpenAIError

from vision.utils.cache import build_cache_key, get_cache_value, set_cache_value

logger = logging.getLogger(__name__)

VISION_MODELS = ["gpt-4o",]
VISION_CACHE_TTL = 1800
DEFAULT_IMAGE_MIME = "image/jpeg"

DATA_URI_PATTERN = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,", re.IGNORECASE)

STRUCTURED_OUTPUT_FORMAT = {
    "type": "json_schema",
    "name": "response_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "food_name": {
                "type": "string",
                "description": "한글 음식 이름. 미확인 시 'Unknown'.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "0과 1 사이의 신뢰도 점수.",
            },
        },
        "required": ["food_name", "confidence"],
        "additionalProperties": False,
    },
}

client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))


def normalize_base64_image(image_base64: str) -> str:
    """Base64 헤더를 제거하고 순수 데이터를 반환."""
    if not image_base64:
        raise ValueError("이미지 데이터가 비어 있습니다.")
    return DATA_URI_PATTERN.sub("", image_base64.strip())


def extract_image_mime_type(image_base64: str) -> str:
    """Base64 데이터 URI에서 MIME 타입을 추출하고 없으면 기본값을 반환."""
    if not image_base64:
        raise ValueError("이미지 데이터가 비어 있습니다.")
    match = DATA_URI_PATTERN.match(image_base64.strip())
    if match:
        return match.group(1).lower()
    return DEFAULT_IMAGE_MIME


def calculate_image_sha(image_base64: str) -> str:
    try:
        decoded = base64.b64decode(image_base64)
    except Exception as exc:  # pragma: no cover - invalid base64
        raise ValueError("유효한 Base64 이미지가 아닙니다.") from exc
    return hashlib.sha256(decoded).hexdigest()


def _strip_code_fences(payload: str) -> str:
    fenced = re.match(r"```(?:json)?\s*(.*?)```", payload, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return payload


def preprocess_api_response(payload: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(payload).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON 파싱 실패: %s", exc)
        raise ValueError("OpenAI 응답을 JSON으로 파싱할 수 없습니다.") from exc
    if "food_name" not in data:
        raise ValueError("OpenAI 응답에 food_name 필드가 없습니다.")
    return data


def _extract_response_text(response: Any) -> str:
    # OpenAI Responses API
    output = getattr(response, "output", None)
    if output:
        first = output[0]
        if hasattr(first, "content") and first.content:
            return first.content[0].text
    # Chat Completions fallback
    choices = getattr(response, "choices", None)
    if choices:
        return choices[0].message.content
    raise ValueError("OpenAI 응답에서 텍스트를 찾을 수 없습니다.")


def _call_openai_vision(
    model: str, prompt: str, image_base64: str, mime_type: str
) -> Dict[str, Any]:
    data_url = f"data:{mime_type};base64,{image_base64}"
    response = client.responses.create(
        model=model,
        temperature=0,
        text={"format": STRUCTURED_OUTPUT_FORMAT},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    text = _extract_response_text(response)
    return preprocess_api_response(text)


def recognize_food_from_image(
    user,
    image_base64: str,
    style_prompt: str,
    cache_timeout: int = VISION_CACHE_TTL,
) -> Tuple[Dict[str, Any], str, bool]:
    mime_type = extract_image_mime_type(image_base64)
    normalized_image = normalize_base64_image(image_base64)
    image_sha = calculate_image_sha(normalized_image)
    cache_key = build_cache_key(user.id, "recognition", image_sha)

    cached = get_cache_value(cache_key)
    if cached:
        return cached, image_sha, True

    prompt = (
        f"{style_prompt}\n\n"
        "업로드된 음식 사진에서 음식 이름을 한글로 하나만 식별하세요. "
        "출력은 JSON 객체 하나이며 스키마는 {\"food_name\": \"string\", \"confidence\": 0.0-1.0} 입니다. "
        "음식을 알 수 없다면 food_name 값을 \"Unknown\"으로 설정하고 confidence는 0.0으로 설정하세요."
    )

    last_error: Exception | None = None
    for model_name in VISION_MODELS:
        try:
            payload = _call_openai_vision(model_name, prompt, normalized_image, mime_type)
            payload["model"] = model_name
            set_cache_value(cache_key, payload, cache_timeout)
            return payload, image_sha, False
        except OpenAIError as exc:
            last_error = exc
            logger.warning("OpenAI Vision 호출 실패 model=%s error=%s", model_name, exc)
        except ValueError as exc:
            last_error = exc
            logger.error("OpenAI Vision 응답 파싱 실패 model=%s error=%s", model_name, exc)
            break

    if last_error:
        raise last_error
    raise RuntimeError("OpenAI Vision 호출에 실패했습니다.")


