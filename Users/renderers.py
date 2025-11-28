try:
    import orjson  # type: ignore
except ImportError:
    orjson = None
from rest_framework.renderers import JSONRenderer

class ORJSONRenderer(JSONRenderer):
    """
    JSON renderer using orjson for faster serialization.
    """
    media_type = 'application/json'
    charset = None
    render_style = 'binary'
    if orjson:
        options = (
            orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SERIALIZE_DATACLASS
            | orjson.OPT_NAIVE_UTC
        )
    else:
        options = 0

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if data is None:
            return b''
        if orjson:
            return orjson.dumps(data, option=self.options)
        return super().render(data, accepted_media_type, renderer_context) 