"""
FORMA chat engine — shared OpenAI streaming + function-calling plumbing.

Used by:
  - /api/chat/guide       (stateless, logged-out, no tools)
  - /api/chat/personal    (logged-in, tool-use, persisted history)
  - /api/chat/plan        (Session 4 — plan-creator, more tools)

The engine streams plain text chunks via an iterator so the Flask SSE
endpoint can forward them verbatim. Tool calls are handled internally:
when the model asks for a tool, we call the dispatcher, wrap the JSON
response in <tool_response> tags to defend against prompt injection,
feed it back to the model, and continue streaming the next turn.

Token usage is tracked per user per day (ExerVisionDB.record_token_usage)
and enforced on entry via check_token_budget.

Model selection is a parameter — Session 4's plan-creator passes gpt-4o
explicitly; the personal chat can auto-downgrade to gpt-4o-mini when the
user has exceeded 50% of their daily budget.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Lazy import — the openai package is optional in dev environments that
# don't yet have OPENAI_API_KEY set. Importing at module load breaks tests
# and keeps the server bootable even without a key.
_openai_module = None
_openai_client = None


def _get_client():
    """Return a cached OpenAI client or raise OpenAIKeyMissing."""
    global _openai_module, _openai_client
    if _openai_client is not None:
        return _openai_client
    if not os.environ.get("OPENAI_API_KEY"):
        raise OpenAIKeyMissing(
            "OPENAI_API_KEY not set. Add it to .env and restart the server."
        )
    if _openai_module is None:
        import openai  # imported here to keep module-import cheap

        _openai_module = openai

    # Explicitly construct a clean httpx client to avoid the
    # `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`
    # error that occurs when OpenAI v1.x tries to forward proxy settings to
    # httpx >= 0.28.0, which dropped the `proxies` parameter in favour of
    # `proxy`. Passing our own http_client bypasses that code path entirely.
    # Proxy configuration (if needed) should be set via the standard
    # HTTP_PROXY / HTTPS_PROXY environment variables, which httpx reads
    # automatically when no explicit client is provided — but since we're
    # supplying one here, set them on the transport directly if present.
    import httpx

    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    http_client = httpx.Client(proxy=proxy_url) if proxy_url else httpx.Client()

    _openai_client = _openai_module.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=http_client,
    )
    return _openai_client


# ── Errors ─────────────────────────────────────────────────────────────


class ChatError(Exception):
    """Base class for chat-layer errors that the SSE endpoint surfaces."""

    code = "chat_error"


class OpenAIKeyMissing(ChatError):
    code = "openai_key_missing"


class TokenBudgetExceeded(ChatError):
    code = "daily_token_budget_exceeded"


class OpenAIUpstreamError(ChatError):
    code = "openai_upstream_error"


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    """In-memory representation of a chat turn.

    `tool_calls` holds the raw tool call list from an assistant turn so the
    engine can re-play multi-step tool dialogs to the model.
    """

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str = ""
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    citations: List[int] = field(default_factory=list)
    name: Optional[str] = None  # tool name for role=="tool"

    def to_openai(self) -> Dict[str, Any]:
        """Shape a message for the OpenAI Chat Completions API."""
        msg: Dict[str, Any] = {"role": self.role}
        if self.role == "tool":
            msg["tool_call_id"] = self.tool_call_id or ""
            # Wrap tool responses in explicit tags — defense against prompt
            # injection via data that smuggles instructions into the tool
            # response string. The model is instructed to treat anything
            # inside these tags as data, not commands.
            msg["content"] = f"<tool_response>{self.content}</tool_response>"
            if self.name:
                msg["name"] = self.name
            return msg
        msg["content"] = self.content or ""
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


# ── Safety helpers ────────────────────────────────────────────────────

MEDICAL_KEYWORDS = (
    "pain", "hurt", "injury", "injured", "torn", "tore", "tear",
    "sprain", "strain", "concussion", "dizzy", "faint", "numb",
    "numbness", "bleeding", "swollen", "swelling", "shooting pain",
    "sharp pain", "can't move", "cant move", "broke", "broken bone",
    "pulled muscle",
)

MEDICAL_DISCLAIMER = (
    "\n\n*If you're experiencing pain or a possible injury, please rest the "
    "affected area and see a qualified healthcare professional — I can't "
    "diagnose or treat injuries.*"
)

USER_MESSAGE_MAX_CHARS = 2000


def sanitize_user_input(text: str) -> str:
    """Trim and hard-cap user-supplied text to prevent megaprompts."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > USER_MESSAGE_MAX_CHARS:
        text = text[:USER_MESSAGE_MAX_CHARS]
    return text


def mentions_medical(text: str) -> bool:
    lowered = (text or "").lower()
    return any(kw in lowered for kw in MEDICAL_KEYWORDS)


# ── Token estimator (cheap fallback) ──────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token count when the upstream response doesn't include usage.

    Matches OpenAI's ~4-chars-per-token heuristic. Close enough for a daily
    budget counter — we just don't want to be off by 10x.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Streaming engine ──────────────────────────────────────────────────


ToolDispatcher = Callable[[str, Dict[str, Any]], Any]


def stream_chat(
    system_prompt: str,
    messages: List[ChatMessage],
    tools: Optional[List[Dict]] = None,
    tool_dispatcher: Optional[ToolDispatcher] = None,
    user_id: Optional[int] = None,
    model: str = "gpt-4o-mini",
    db=None,  # ExerVisionDB — injected by the endpoint for usage tracking
    max_tool_iterations: int = 4,
    append_medical_disclaimer: bool = False,
    on_complete: Optional[Callable[[str, List[int]], None]] = None,
) -> Iterator[str]:
    """Stream an assistant response as plain text chunks.

    Yields raw text tokens. Tool calls are handled transparently — when the
    model asks for tools, the engine calls the dispatcher, feeds results
    back, and resumes streaming. Citations collected from tool responses
    (session ids referenced by get_session_detail or get_recent_sessions)
    are passed to `on_complete` along with the full assistant text.

    Raises ChatError subclasses on policy/budget/upstream failures. The SSE
    endpoint catches those and emits a structured error event.
    """
    # ── Budget gate ─────────────────────────────────────────────────
    if user_id is not None and db is not None:
        budget = db.check_token_budget(user_id)
        if not budget["allowed"]:
            raise TokenBudgetExceeded(
                f"Daily chat token budget of {budget['budget']} exceeded "
                f"(used {budget['used']})."
            )

    client = _get_client()  # raises OpenAIKeyMissing if no key

    # Assemble the OpenAI messages list starting with system prompt.
    oai_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    for m in messages:
        oai_messages.append(m.to_openai())

    collected_text_parts: List[str] = []
    collected_citations: List[int] = []
    iteration = 0

    # Tool-use loop — each iteration is one OpenAI completion call. If the
    # model requests tools, we dispatch them and go around again.
    while iteration < max_tool_iterations:
        iteration += 1

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=oai_messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                stream=True,
                temperature=0.7,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("OpenAI stream open failed")
            raise OpenAIUpstreamError(f"OpenAI error: {e}") from e

        # Accumulate deltas for this round.
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        text_this_round: List[str] = []
        finish_reason: Optional[str] = None

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # Text content — yield immediately so the client sees tokens
                # as they arrive.
                if getattr(delta, "content", None):
                    piece = delta.content
                    text_this_round.append(piece)
                    collected_text_parts.append(piece)
                    yield piece

                # Tool call deltas — accumulate per index.
                tc_delta = getattr(delta, "tool_calls", None)
                if tc_delta:
                    for tc in tc_delta:
                        idx = tc.index
                        slot = tool_calls_acc.setdefault(
                            idx,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        if getattr(tc, "function", None):
                            fn = tc.function
                            if getattr(fn, "name", None):
                                slot["function"]["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                slot["function"]["arguments"] += fn.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except Exception as e:  # noqa: BLE001
            logger.exception("OpenAI stream read failed")
            raise OpenAIUpstreamError(f"OpenAI stream error: {e}") from e

        # Decide what to do next based on the finish reason.
        if finish_reason == "tool_calls" and tool_calls_acc and tool_dispatcher:
            # Push the assistant tool-call turn, then the tool results.
            tool_calls_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
            oai_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_list,
                }
            )

            for tc in tool_calls_list:
                fn_name = tc["function"].get("name", "")
                raw_args = tc["function"].get("arguments", "") or "{}"
                try:
                    args_dict = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args_dict = {}
                try:
                    result = tool_dispatcher(fn_name, args_dict)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Tool dispatch %s failed: %s", fn_name, e)
                    result = {"error": "tool_failed", "detail": str(e)[:200]}

                # Harvest session-id citations from known tool shapes.
                collected_citations.extend(_extract_citations(fn_name, result))

                tool_content = json.dumps(result, default=str)[:8000]
                oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": f"<tool_response>{tool_content}</tool_response>",
                    }
                )

            # Loop again so the model can use the tool results.
            continue

        # Normal finish — we're done.
        break

    else:  # while iteration exhausted
        logger.warning("Chat tool loop hit max iterations (%d)", max_tool_iterations)

    assistant_text = "".join(collected_text_parts)

    # Medical disclaimer append — only if the assistant didn't already say
    # "healthcare professional" (avoid double-disclaimer loops).
    if append_medical_disclaimer and "professional" not in assistant_text.lower():
        yield MEDICAL_DISCLAIMER
        assistant_text += MEDICAL_DISCLAIMER

    # Record token usage — best-effort estimate since streaming doesn't
    # report usage deltas. We compute input-token cost from the assembled
    # messages once, output-token cost from the accumulated assistant text.
    if user_id is not None and db is not None:
        input_tokens = sum(
            _estimate_tokens(
                str(m.get("content", "")) if isinstance(m, dict) else str(m)
            )
            for m in oai_messages
        )
        output_tokens = _estimate_tokens(assistant_text)
        try:
            db.record_token_usage(user_id, input_tokens, output_tokens)
        except Exception as e:  # noqa: BLE001
            logger.debug("Token usage write failed: %s", e)

    # De-duplicated citations, preserving order.
    seen = set()
    unique_citations: List[int] = []
    for cid in collected_citations:
        if cid in seen:
            continue
        seen.add(cid)
        unique_citations.append(cid)

    if on_complete is not None:
        try:
            on_complete(assistant_text, unique_citations)
        except Exception as e:  # noqa: BLE001
            logger.debug("on_complete callback failed: %s", e)


def _extract_citations(fn_name: str, result: Any) -> List[int]:
    """Pull session ids out of common tool-response shapes."""
    if not isinstance(result, (dict, list)):
        return []
    out: List[int] = []

    def _add(val: Any) -> None:
        try:
            out.append(int(val))
        except (TypeError, ValueError):
            pass

    if isinstance(result, dict):
        if "session_id" in result:
            _add(result["session_id"])
        if "id" in result and fn_name in (
            "get_session_detail",
        ):
            _add(result["id"])
        if "session_ids" in result and isinstance(result["session_ids"], list):
            for v in result["session_ids"]:
                _add(v)
        # get_recent_sessions returns {sessions: [{id, ...}]}
        if "sessions" in result and isinstance(result["sessions"], list):
            for row in result["sessions"]:
                if isinstance(row, dict) and "id" in row:
                    _add(row["id"])
    elif isinstance(result, list):
        for row in result:
            if isinstance(row, dict):
                if "id" in row:
                    _add(row["id"])
                elif "session_id" in row:
                    _add(row["session_id"])
    return out
