from rest_framework.parsers import JSONParser
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import orjson
else:
    try:
        import orjson
    except ImportError:
        orjson: Optional[Any] = None

class ORJSONParser(JSONParser):
    """
    JSON parser using orjson for faster deserialization.
    """
    media_type = 'application/json'

    def parse(self, stream, media_type=None, parser_context=None):
        data_bytes = stream.read()
        if not data_bytes:
            return {}
        if orjson:
            return orjson.loads(data_bytes)  # type: ignore[attr-defined]
        return super().parse(stream, media_type, parser_context)
