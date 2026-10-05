"""Bounded HTTP request semantics; acquisition uses recorded observations."""
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def specification(data):
    spec = json.loads(data)
    if not isinstance(spec, dict) or set(spec) - {"url", "method", "headers", "body", "timeout"}:
        raise ValueError("HTTP request accepts url/method/headers/body/timeout")
    url = spec.get("url")
    if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).hostname:
        raise ValueError("HTTP requires an http/https URL")
    timeout = spec.get("timeout", 5)
    if type(timeout) not in (int, float) or not 0 < timeout <= 30:
        raise ValueError("HTTP timeout must be in (0,30]")
    method = spec.get("method", "GET")
    if method not in ("GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"):
        raise ValueError("unsupported HTTP method")
    headers = spec.get("headers", {})
    if not isinstance(headers, dict) or any(not isinstance(k, str) or not k or any(c in k for c in "\r\n\0") or not isinstance(v, str) or any(c in v for c in "\r\n\0") for k, v in headers.items()):
        raise ValueError("invalid HTTP headers")
    body = spec.get("body")
    if body is not None and (not isinstance(body, str) or len(body) % 2 or any(c not in "0123456789abcdefABCDEF" for c in body)):
        raise ValueError("HTTP body requires hexadecimal bytes")
    return Request(url, data=bytes.fromhex(body) if body is not None else None, headers=headers, method=method), timeout


def decode(data):
    result = json.loads(data)
    if not isinstance(result, dict) or set(result) != {"status", "body"} or type(result["status"]) is not int or not 100 <= result["status"] <= 599:
        raise ValueError("invalid HTTP result")
    body = result["body"]
    if not isinstance(body, str) or len(body) % 2 or any(c not in "0123456789abcdefABCDEF" for c in body):
        raise ValueError("HTTP result body requires byte hex")
    return result


def run(host, data):
    request, timeout = specification(data)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        response = error  # Error statuses retain their bodies for explicit handling.
    with response:
        body = response.read(host.max_buffer_bytes // 2 + 1)
        if len(body) > host.max_buffer_bytes // 2:
            raise ValueError("HTTP response exceeds byte limit")
        return json.dumps({"status": response.code, "body": body.hex()}, sort_keys=True, separators=(",", ":")).encode()
