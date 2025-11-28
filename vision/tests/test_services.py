from django.test import SimpleTestCase

from vision.services.openai_vision import normalize_base64_image, preprocess_api_response
from vision.utils.cache import build_cache_key


class OpenAIVisionUtilsTests(SimpleTestCase):
    def test_preprocess_api_response_strips_code_fence(self):
        payload = "```json\n{\"food_name\": \"사과\", \"confidence\": 0.92}\n```"
        result = preprocess_api_response(payload)
        self.assertEqual(result["food_name"], "사과")

    def test_normalize_base64_image_removes_prefix(self):
        base64_value = "data:image/png;base64,ZmFrZS1kYXRh"
        normalized = normalize_base64_image(base64_value)
        self.assertEqual(normalized, "ZmFrZS1kYXRh")


class CacheKeyTests(SimpleTestCase):
    def test_cache_key_uses_namespace_and_user(self):
        key = build_cache_key(10, "recognition", "abcd")
        self.assertTrue(key.startswith("vision:recognition:10:"))


