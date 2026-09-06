import io
import json
from urllib.error import HTTPError

from shopping_grpo.evaluation.model_client import _http_error_message


def _error(body: bytes, *, reason: str = "Bad Request") -> HTTPError:
    return HTTPError(
        "https://provider.example/v1/chat/completions",
        400,
        reason,
        {},
        io.BytesIO(body),
    )


def test_http_error_message_extracts_only_provider_code_and_message():
    body = json.dumps(
        {
            "error": {
                "code": "invalid_parameter",
                "message": "thinking is unsupported",
                "request": {"authorization": "must-not-appear"},
            }
        }
    ).encode()

    message = _http_error_message(_error(body))

    assert message == "invalid_parameter: thinking is unsupported"
    assert "authorization" not in message


def test_http_error_message_does_not_echo_non_json_response_body():
    message = _http_error_message(_error(b"proxy rejected request with details"))

    assert message == "Bad Request"
