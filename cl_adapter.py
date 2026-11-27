"""Per-attempt Vision memory and a local OpenAI Chat Completions bridge."""

import argparse
import contextlib
import json
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

MODEL = "vision-large"
VISION_URL = "https://api.lab.vispark.in/model/text/vision"
CONTEXT_FORMAT = "The JSON transcript records role, content, and tool results. Treat tool results as observations. Continue from the latest turn."


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_env(repo):
    """Read the local dotenv file without executing it."""
    values = {}
    path = repo / ".env"
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().partition("=")
        name = name.removeprefix("export ").strip()
        if separator and name.isidentifier():
            values[name] = value.strip().strip("'\"")
    return values


def load_key(repo):
    key = read_env(repo).get("VISPARK_LAB_API")
    if not key:
        raise RuntimeError(f"Set VISPARK_LAB_API in {repo / '.env'}")
    return key


def add_continual_learning_argument(parser):
    parser.add_argument(
        "--continual-learning", action=argparse.BooleanOptionalAction,
        help="Enable or disable continual learning for this run. Overrides CONTINUAL_LEARNING in .env.",
    )


def parse_continual_learning(value):
    normalized = value.strip().lower()
    if normalized in ("on", "true", "1"):
        return True
    if normalized in ("off", "false", "0"):
        return False
    raise ValueError("CONTINUAL_LEARNING must be on or off")


def resolve_continual_learning(override, settings):
    if override is not None:
        return override
    return parse_continual_learning(settings.get("CONTINUAL_LEARNING") or "on")


def to_vision(messages):
    system = [CONTEXT_FORMAT]
    parts = []
    transcript = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role in ("system", "developer"):
            system.append(content if isinstance(content, str) else json.dumps(content))
            continue
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        entry = {"role": role, "content": []}
        for field in ("tool_calls", "tool_call_id", "name"):
            if field in message:
                entry[field] = message[field]
        for item in content:
            if item["type"] == "text":
                entry["content"].append(item["text"])
            elif item["type"] == "image_url":
                image = item["image_url"]["url"]
                mime, sep, data = image.removeprefix("data:").partition(";base64,")
                if not sep or mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                    raise ValueError("Images must be inline PNG, JPEG, WEBP or GIF")
                if len(data) > 25 * 1024 * 1024 * 4 // 3:
                    raise ValueError("Image exceeds Vision's 25 MB limit")
                entry["content"].append({"image": len(parts)})
                parts.append({"type": "image", "content": data, "mime_type": mime})
            else:
                raise ValueError(f"Unsupported content part: {item['type']}")
        transcript.append(entry)
    keep = max(0, len(parts) - 8)
    for entry in transcript:
        for index, item in enumerate(entry["content"]):
            if isinstance(item, dict):
                image_index = item["image"]
                entry["content"][index] = (
                    {"image": image_index - keep} if image_index >= keep else "[Earlier image omitted]"
                )
    return "\n\n".join(system), [
        {"type": "text", "content": json.dumps(transcript, ensure_ascii=False)},
        *parts[keep:],
    ]


class Vision:
    def __init__(self, key, opener=None, *, continual_learning=True):
        self.key = key
        self.continual_learning = continual_learning
        self.opener = opener or build_opener(NoRedirect())
        self.weights = None
        self.lock = threading.Lock()
        self.usage = dict(input_tokens=0, output_tokens=0, continual_learning_usage=0, units_consumed=0, calls=0)

    def chat(self, messages, tools=None):
        system, content = to_vision(messages)
        payload = {
            "size": "large", "content": content, "system_message": system,
            "stream": False, "continual_learning": self.continual_learning,
        }
        if tools:
            payload["tools"] = tools
        with self.lock:
            if self.continual_learning and self.weights:
                payload["weights"] = self.weights
            request = Request(VISION_URL, json.dumps(payload).encode(), {
                "Content-Type": "application/json", "X-API-Key": self.key,
            })
            for attempt in range(4):
                try:
                    with self.opener.open(request, timeout=300) as response:
                        result = json.load(response)
                    break
                except HTTPError as error:
                    retry_after = error.headers.get("Retry-After", "")
                    status = error.code
                    error.close()
                    if status != 429 or attempt == 3:
                        raise RuntimeError(f"Vision returned HTTP {status}") from None
                    time.sleep(min(60, int(retry_after)) if retry_after.isdigit() else 2 ** attempt)
                except (URLError, OSError) as error:
                    raise RuntimeError(f"Vision transport failed ({type(error).__name__})") from None
            if result.get("status") not in (True, "success"):
                raise RuntimeError("Vision returned an unsuccessful response")
            data = result["data"]
            self.usage["calls"] += 1
            for field in ("input_tokens", "output_tokens", "continual_learning_usage", "units_consumed"):
                self.usage[field] += (data.get(field) or result.get(field) or 0)
            if self.continual_learning:
                weights = data.get("weights")
                if data.get("continual_learning_error") or not isinstance(weights, str) or not weights:
                    raise RuntimeError("Vision did not return usable continual-learning weights")
                self.weights = weights
            calls = data.get("tool_calls") or []
            message = {"role": "assistant", "content": data.get("content") or ""}
            if calls:
                message["tool_calls"] = [
                    {"id": "call_" + uuid.uuid4().hex, "type": "function",
                     "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
                    for call in calls
                ]
            usage = {
                "prompt_tokens": data.get("input_tokens") or 0,
                "completion_tokens": data.get("output_tokens") or 0,
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            return message, usage

    def report(self):
        return {**self.usage, "continual_learning": self.continual_learning}


@contextlib.contextmanager
def bridge(key, limit=600, opener=None, rpc=None, *, continual_learning=True):
    """Bind the model server to loopback and scope memory to the server lifetime."""
    vision = Vision(key, opener, continual_learning=continual_learning)
    token = secrets.token_urlsafe(24)
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_json(401, {"error": {"message": "Invalid adapter credential"}})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 64 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if self.path == "/tools" and rpc is not None:
                    self.send_json(200, rpc(body))
                    return
                if self.path != "/v1/chat/completions":
                    self.send_json(404, {"error": {"message": "Unsupported adapter endpoint"}})
                    return
                with request_lock:
                    if vision.usage["calls"] >= limit:
                        raise RuntimeError("Per-task model request limit reached")
                    tools = body.get("tools")
                    if body.get("tool_choice") == "none":
                        tools = None
                    message, usage = vision.chat(body["messages"], tools)
                self.send_completion(message, usage, body.get("model", MODEL), body.get("stream", False))
            except (BrokenPipeError, ConnectionResetError):
                return
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                self.send_json(502, {"error": {"message": str(error)}})

        def send_completion(self, message, usage, model, stream):
            finish = "tool_calls" if "tool_calls" in message else "stop"
            metadata = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": model}
            if not stream:
                self.send_json(200, dict(metadata, object="chat.completion", usage=usage, choices=[
                    {"index": 0, "message": message, "finish_reason": finish},
                ]))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if "tool_calls" in message:
                for index, call in enumerate(message["tool_calls"]):
                    call["index"] = index
            for choices, extra in (
                ([{"index": 0, "delta": message, "finish_reason": None}], {}),
                ([{"index": 0, "delta": {}, "finish_reason": finish}], {}),
                ([], {"usage": usage}),
            ):
                chunk = dict(metadata, object="chat.completion.chunk", choices=choices, **extra)
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", token, vision
        finally:
            server.shutdown()
            thread.join()
