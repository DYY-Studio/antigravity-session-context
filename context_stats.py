#!/usr/bin/env python3
"""Read-only context window and compaction budget inspector for Antigravity conversation .db files."""

import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

STEP_USER_INPUT = 19
STEP_PLANNER_RESPONSE = 20
STEP_CHECKPOINT = 30
STEP_GENERIC = 140
GENERIC_TYPE = 132
ORIGINAL_STEP_TYPE_URL = "/gemini_coder.Step"

STATUS_PENDING = 1
STATUS_RUNNING = 2
STATUS_DONE = 3
STATUS_INVALID = 4
STATUS_CANCELED = 6
STATUS_ERROR = 7
STATUS_GENERATING = 8
STATUS_WAITING = 9
STATUS_INTERRUPTED = 10
STATUS_HALTED = 11

TERM_REASON_NAMES = {
    1: "error",
    2: "user_canceled",
    3: "max_invocations",
    4: "no_tool_call",
    5: "max_forced_invocations",
    6: "early_continue",
    7: "terminal_step_type",
    8: "terminal_custom_hook",
    9: "injected_response",
    10: "max_token_budget_exceeded",
    11: "halted_step",
}

TOKEN_SYSTEM_PROMPT, TOKEN_TOOLS, TOKEN_RULES, TOKEN_MCP_TOOLS, TOKEN_SKILLS = 1, 2, 3, 4, 5
TOKEN_SUBAGENTS, TOKEN_CHAT_MESSAGES, TOKEN_WORKFLOWS, TOKEN_PLUGINS, TOKEN_FLOWS = 6, 7, 8, 9, 10

STEP_TYPES = {
    "UNSPECIFIED": 0, "DUMMY": 1, "FINISH": 2, "PLAN_INPUT": 3, "MQUERY": 4, "CODE_ACTION": 5,
    "GIT_COMMIT": 6, "GREP_SEARCH": 7, "VIEW_FILE": 8, "LIST_DIRECTORY": 9, "COMPILE": 10,
    "VIEW_CODE_ITEM": 13, "USER_INPUT": 14, "PLANNER_RESPONSE": 15, "ERROR_MESSAGE": 17,
    "RUN_COMMAND": 21, "CHECKPOINT": 23, "PROPOSE_CODE": 24, "FIND": 25, "SUGGESTED_RESPONSES": 27,
    "COMMAND_STATUS": 28, "MEMORY": 29, "READ_URL_CONTENT": 31, "VIEW_CONTENT_CHUNK": 32,
    "SEARCH_WEB": 33, "RETRIEVE_MEMORY": 34, "MCP_TOOL": 38, "MANAGER_FEEDBACK": 39,
    "TOOL_CALL_PROPOSAL": 40, "TOOL_CALL_CHOICE": 41, "TRAJECTORY_CHOICE": 42, "CLIPBOARD": 45,
    "VIEW_FILE_OUTLINE": 47, "POST_PR_REVIEW": 49, "LIST_RESOURCES": 51, "READ_RESOURCE": 52,
    "LINT_DIFF": 53, "FIND_ALL_REFERENCES": 54, "BRAIN_UPDATE": 55, "OPEN_BROWSER_URL": 56,
    "RUN_EXTENSION_CODE": 57, "PROPOSAL_FEEDBACK": 59, "TRAJECTORY_SEARCH": 60,
    "EXECUTE_BROWSER_JAVASCRIPT": 61, "LIST_BROWSER_PAGES": 62, "CAPTURE_BROWSER_SCREENSHOT": 63,
    "CLICK_BROWSER_PIXEL": 64, "READ_TERMINAL": 65, "CAPTURE_BROWSER_CONSOLE_LOGS": 66,
    "READ_BROWSER_PAGE": 67, "BROWSER_GET_DOM": 68, "CODE_SEARCH": 73, "BROWSER_INPUT": 74,
    "BROWSER_MOVE_MOUSE": 75, "BROWSER_SELECT_OPTION": 76, "BROWSER_SCROLL_UP": 77,
    "BROWSER_SCROLL_DOWN": 78, "BROWSER_CLICK_ELEMENT": 79, "BROWSER_PRESS_KEY": 80,
    "TASK_BOUNDARY": 81, "NOTIFY_USER": 82, "CODE_ACKNOWLEDGEMENT": 83, "INTERNAL_SEARCH": 84,
    "BROWSER_SUBAGENT": 85, "FILE_CHANGE": 86, "MOVE": 87, "BROWSER_SCROLL": 88,
    "KNOWLEDGE_GENERATION": 89, "EPHEMERAL_MESSAGE": 90, "GENERATE_IMAGE": 91,
    "DELETE_DIRECTORY": 92, "COMPILE_APPLET": 93, "INSTALL_APPLET_DEPENDENCIES": 94,
    "INSTALL_APPLET_PACKAGE": 95, "BROWSER_RESIZE_WINDOW": 96, "BROWSER_DRAG_PIXEL_TO_PIXEL": 97,
    "CONVERSATION_HISTORY": 98, "KNOWLEDGE_ARTIFACTS": 99, "SEND_COMMAND_INPUT": 100,
    "SYSTEM_MESSAGE": 101, "WAIT": 102, "AGENCY_TOOL_CALL": 103, "CIDER_AGENT_DUMMY": 104,
    "BUILD_CLEANER": 105, "BLAZE_BUILD_TARGETS": 106, "BLAZE_TEST_TARGETS": 107,
    "SET_UP_FIREBASE": 108, "MOMA": 109, "RESTART_DEV_SERVER": 110, "DEPLOY_FIREBASE": 111,
    "SHELL_EXEC": 112, "BROWSER_MOUSE_WHEEL": 113, "LINT_APPLET": 114, "KI_INSERTION": 116,
    "RETRIEVE_CONTENT": 117, "CRITIQUE": 118, "FINDINGS": 119, "BROWSER_MOUSE_UP": 120,
    "BROWSER_MOUSE_DOWN": 121, "WORKSPACE_API": 122, "BROWSER_LIST_NETWORK_REQUESTS": 123,
    "BROWSER_GET_NETWORK_REQUEST": 124, "BROWSER_REFRESH_PAGE": 125, "EDIT_NOTEBOOK": 126,
    "INVOKE_SUBAGENT": 127, "WRITE_BLOB": 128, "READ_NOTEBOOK": 129, "PROPOSE_AI_COMMENTS": 130,
    "START_CODE_REVIEW": 131, "GENERIC": 132, "SET_UP_CLOUD_SQL": 133, "EXECUTE_NOTEBOOK": 134,
    "CLOUD_SQL_UPDATE_SCHEMA": 135, "RPC_ACTION": 136, "CLOUD_SQL_EXECUTE_SQL": 137,
    "ASK_QUESTION": 138, "DIRECTORY_RULES": 139, "TOOL_SEARCH": 140,
}

CAT_USER, CAT_AGENT, CAT_TOOL, CAT_CHECKPOINT, CAT_SYSTEM = 0, 1, 2, 3, 4

CATEGORIES = [
    ("User messages", "user", True),
    ("Agent responses", "agent", True),
    ("Tool calls", "tool", True),
    ("System prompt", "system_prompt", False),
    ("System tools", "system_tools", False),
    ("System overhead", "system_overhead", False),
    ("Rules", "rules", False),
    ("Skills", "skills", False),
    ("Subagents", "subagents", False),
    ("Legacy workflows", "legacy_workflows", False),
    ("Workflows", "workflows", False),
    ("Plugins", "plugins", False),
    ("MCP tools", "mcp_tools", False),
    ("System messages", "system_messages", False),
]

NATIVE_CONTEXT_WINDOWS = [
    ("claude-opus-4-6", 250_000),
    ("claude-sonnet-4-6", 250_000),
    ("claude-", 1_000_000),
    ("gpt-oss-", 131_072),
    ("gemini-", 1_048_576),
]

EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max", "tiered")
WORD_LABELS = {"gpt": "GPT", "oss": "OSS"}

ENUM_MODEL_IDS = {
    1417: "gemini-4-argon-high",
    1418: "gemini-4-argon-medium",
    1419: "gemini-4-argon-low",
    1402: "claude-opus-5-5-high",
    1401: "claude-opus-5-5-medium",
    1400: "claude-opus-5-5-low",
    1405: "claude-sonnet-5-5-high",
    1404: "claude-sonnet-5-5-medium",
    1403: "claude-sonnet-5-5-low",
    1318: "gemini-3.8-flash-high",
    1319: "gemini-3.8-flash-medium",
    1320: "gemini-3.8-flash-low",
    1322: "gemini-3.8-flash-tiered",
    1198: "gemini-3.5-flash-lite",
    1050: "gemini-3.1-flash-lite",
    1021: "gemini-3.1-flash-image",
    1018: "gemini-3-flash",
}

SUBAGENT_CID_RE = re.compile(r'"conversationId"\s*:\s*"([0-9a-fA-F-]{36})"')


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7
    raise ValueError("Truncated varint")


_GEN_FIELDS = frozenset((1, 3))
_CHAT_FIELDS = frozenset((4, 9, 19, 21))
_PLANNER_FIELDS = frozenset((28,))
_MD_USAGE_FIELDS = frozenset((9,))
_MD_TC_FIELDS = frozenset((4,))


class Msg:
    """Lazily decoded protobuf message: field number -> list of raw values."""

    def __init__(self, buf: bytes | None, wanted: frozenset[int] | None = None):
        self.fields: dict[int, list[Any]] = {}
        self._msg_cache: dict[tuple[int, frozenset[int] | None], Msg | None] = {}
        raw = buf or b""
        pos, n = 0, len(raw)
        while pos < n:
            try:
                b = raw[pos]
                if b < 0x80:
                    tag = b
                    pos += 1
                else:
                    tag, pos = read_varint(raw, pos)
                fn, wt = tag >> 3, tag & 7
                keep = wanted is None or fn in wanted
                if wt == 0:
                    if pos < n and raw[pos] < 0x80:
                        val: Any = raw[pos]
                        pos += 1
                    else:
                        val, pos = read_varint(raw, pos)
                elif wt == 2:
                    if pos < n and raw[pos] < 0x80:
                        ln = raw[pos]
                        pos += 1
                    else:
                        ln, pos = read_varint(raw, pos)
                    if ln < 0 or pos + ln > n:
                        break
                    if keep:
                        val = raw[pos : pos + ln]
                    pos += ln
                elif wt == 1:
                    if pos + 8 > n:
                        break
                    if keep:
                        val = raw[pos : pos + 8]
                    pos += 8
                elif wt == 5:
                    if pos + 4 > n:
                        break
                    if keep:
                        val = raw[pos : pos + 4]
                    pos += 4
                else:
                    break
            except ValueError:
                break
            if keep:
                self.fields.setdefault(fn, []).append(val)

    def has(self, fn: int) -> bool:
        return fn in self.fields

    def int(self, fn: int, signed32: bool = False) -> int:
        v = self.fields.get(fn, [0])[-1]
        v = v if isinstance(v, int) else 0
        if signed32 and v >= 1 << 63:
            v -= 1 << 64
        return v

    def bytes(self, fn: int) -> bytes:
        v = self.fields.get(fn, [b""])[-1]
        return v if isinstance(v, bytes) else b""

    def str(self, fn: int) -> str:
        return self.bytes(fn).decode("utf-8", errors="replace")

    def msg(self, fn: int, wanted: frozenset[int] | None = None) -> "Msg | None":
        key = (fn, wanted)
        if key in self._msg_cache:
            return self._msg_cache[key]
        res = Msg(self.bytes(fn), wanted) if self.has(fn) else None
        self._msg_cache[key] = res
        return res

    def msgs(self, fn: int, wanted: frozenset[int] | None = None) -> list["Msg"]:
        return [Msg(v, wanted) for v in self.fields.get(fn, []) if isinstance(v, bytes)]


def fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000.0
        return f"{v:.0f}M" if v.is_integer() else f"{v:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000.0:.1f}k"
    return str(n)


def pct(n: int, limit: int) -> float:
    return n / limit * 100.0 if limit else 0.0


def go_div4(n: int) -> int:
    return int(n / 4)


def step_category(step_type: int) -> int:
    if step_type == 14:
        return CAT_USER
    if step_type == 15:
        return CAT_AGENT
    if step_type == 23:
        return CAT_CHECKPOINT
    if step_type in (0, 2, 101):
        return CAT_SYSTEM
    return CAT_TOOL


def packed_original_step(generic: Msg) -> bytes | None:
    result = generic.msg(2)
    if result is None:
        return None
    payload = result.msg(6)
    if payload is not None and payload.str(1).endswith(ORIGINAL_STEP_TYPE_URL):
        return payload.bytes(2)
    for entry in result.msgs(5):
        if entry.str(1) == "__original_cortex_step":
            value = entry.msg(2)
            if value is not None and value.str(1).endswith(ORIGINAL_STEP_TYPE_URL):
                return value.bytes(2)
    return None


class Step:
    def __init__(self, pos: int, payload: bytes, step_type: int, status: int, metadata: bytes | None):
        self.pos = pos
        self.pb = Msg(payload)
        self.type = self.pb.int(1) if self.pb.has(1) else step_type
        self.status = self.pb.int(4) if self.pb.has(4) else status
        self.md = self.pb.msg(5) or Msg(metadata)
        self.generic_pb = self.pb.msg(STEP_GENERIC) if self.type == GENERIC_TYPE else None
        if self.type == GENERIC_TYPE and self.generic_pb is not None:
            original = packed_original_step(self.generic_pb)
            if original is not None:
                self.pb = Msg(original)
                self.type = self.pb.int(1)
                self.md = self.pb.msg(5) or Msg(b"")

    def payload(self, fn: int) -> Msg | None:
        return self.pb.msg(fn)


def estimate_step_tokens(step: Step) -> int:
    ui = step.payload(STEP_USER_INPUT)
    pr = step.payload(STEP_PLANNER_RESPONSE)
    if ui is not None:
        n = len(ui.bytes(2))
    elif pr is not None:
        n = len(pr.bytes(1))
    else:
        tc = step.md.msg(4)
        n = len(tc.bytes(3)) if tc else 0
        out = step.md.int(23, signed32=True)
        if out > 0:
            return out
    if n == 0:
        return 0
    return max(1, go_div4(n))


def step_tokens(step: Step) -> int:
    usage = step.md.msg(9)
    if usage is not None:
        return usage.int(3)
    return estimate_step_tokens(step)


def post_turn_step_tokens(step: Step) -> int:
    base = step_tokens(step)
    if step.md.int(23, signed32=True) <= 0 and step.generic_pb is not None:
        res = step.generic_pb.msg(2)
        if res is not None:
            base += go_div4(len(res.bytes(1)))
    return base


def effective_budget_tokens(steps: list[Step], st: dict[str, Any]) -> int:
    if st.get("compacted"):
        return int(st["total"])
    ckpt_pos = -1
    for i, s in enumerate(steps):
        if s.payload(STEP_CHECKPOINT) is not None and s.status == STATUS_DONE:
            ckpt_pos = i
    pos_last = -1
    prefix_tok = 0
    for i in range(len(steps) - 1, ckpt_pos, -1):
        s = steps[i]
        if s.payload(STEP_PLANNER_RESPONSE) is None:
            continue
        u = s.md.msg(9)
        if u is None:
            continue
        inp, out_tok, cache_r, cache_w = u.int(2), u.int(3), u.int(5), u.int(6)
        if inp + cache_r + cache_w > 0:
            pos_last = i
            prefix_tok = inp + cache_r + cache_w + out_tok
            break
    if pos_last >= 0:
        delta_tok = sum(
            post_turn_step_tokens(s)
            for s in steps[pos_last + 1 :]
            if s.payload(STEP_CHECKPOINT) is None
        )
        return max(int(st["total"]), prefix_tok + delta_tok)
    return int(st["total"])


def compaction_tokens(step: Step) -> int:
    for ri in reversed(step.md.msgs(28)):
        usage = ri.msg(2)
        if usage is None:
            continue
        resp = usage.int(10)
        if resp:
            return resp
        out, thinking = usage.int(3), usage.int(9)
        return out - thinking if thinking < out else 0
    return 0


def checkpoint_summary_tokens(step: Step) -> int:
    t = compaction_tokens(step)
    if t:
        return t
    ck = step.payload(STEP_CHECKPOINT) or Msg(b"")
    return go_div4(len(ck.bytes(4)) + len(ck.bytes(5)) + len(ck.bytes(6)))


def checkpoint_newer_than_generation(steps: list[Step], chat_start: Msg | None) -> tuple[int, Step | None]:
    i = len(steps) - 1
    while i >= 0:
        s = steps[i]
        if s.payload(STEP_CHECKPOINT) is not None and s.status == STATUS_DONE:
            break
        i -= 1
    if i < 0:
        return -1, None
    for s in steps[i + 1 :]:
        if s.payload(STEP_PLANNER_RESPONSE) is not None and s.status == STATUS_DONE:
            return -1, None
    gen_ck = chat_start.int(2, signed32=True) if chat_start is not None else -1
    ck_pb = steps[i].payload(STEP_CHECKPOINT)
    ck_idx = ck_pb.int(1) if ck_pb is not None else 0
    if gen_ck >= ck_idx:
        return -1, None
    return i, steps[i]


def chat_tokens_by_category(items: list[Msg]) -> tuple[bool, list[int]]:
    out = [0, 0, 0, 0]
    if not items:
        return False, out
    for it in items:
        left, sep, right = it.str(1).partition(": ")
        if not sep or right not in STEP_TYPES:
            return False, [0, 0, 0, 0]
        digits = left[1:] if left[:1] in "+-" else left
        if not digits or not digits.isascii() or not digits.isdigit():
            return False, [0, 0, 0, 0]
        cat = step_category(STEP_TYPES[right])
        tok = it.int(3, signed32=True)
        if cat == CAT_USER:
            out[0] += tok
        elif cat in (CAT_AGENT, CAT_CHECKPOINT):
            out[1] += tok
        elif cat == CAT_TOOL:
            out[2] += tok
        else:
            out[3] += tok
    return True, out


def _gen_chat(gen: Msg | None) -> Msg | None:
    return gen.msg(1, _CHAT_FIELDS) if gen is not None else None


def collect_stats(
    steps: list[Step],
    gen: Msg | None,
    limit: int,
    newer_override: tuple[int, Step | None] | None = None,
) -> dict[str, Any]:
    st: dict[str, Any] = {
        k: 0
        for k in (
            "user", "agent", "tool", "checkpoint", "system_messages", "system_prompt", "system_tools",
            "system_overhead", "rules", "skills", "subagents", "legacy_workflows", "workflows",
            "plugins", "mcp_tools",
        )
    }
    st["has_breakdown"] = False
    st["active_checkpoint"] = -1
    checkpoints: list[dict[str, Any]] = []

    chat = _gen_chat(gen)
    chat_start = chat.msg(9) if chat is not None else None
    if newer_override is not None:
        newer_pos, newer_step = newer_override
    else:
        newer_pos, newer_step = checkpoint_newer_than_generation(steps, chat_start)
    has_newer = newer_step is not None
    st["compacted"] = has_newer

    for pos, step in enumerate(steps):
        cat = step_category(step.type)
        tok = step_tokens(step)
        counted = 0 if has_newer and newer_pos > pos else tok
        if cat == CAT_USER:
            st["user"] += counted
        elif cat == CAT_AGENT:
            st["agent"] += counted
        elif cat == CAT_TOOL:
            st["tool"] += counted
        elif cat == CAT_CHECKPOINT:
            st["checkpoint"] += tok
            ck = step.payload(STEP_CHECKPOINT)
            if ck is not None:
                checkpoints.append({
                    "index": ck.int(1),
                    "start": ck.int(11),
                    "end": ck.int(12),
                    "intent": ck.str(4),
                })
        else:
            st["system_messages"] += counted

    cwm = chat_start.msg(10) if chat_start is not None else None
    breakdown = cwm.msg(3) if cwm is not None else None
    if breakdown is not None:
        st["has_breakdown"] = True
        seen_prompt = seen_tools = False
        chat_tok = 0
        chat_items: list[Msg] = []
        for g in breakdown.msgs(1):
            gtype, gtok = g.int(2), g.int(4, signed32=True)
            if gtype == TOKEN_SYSTEM_PROMPT:
                extracted = 0
                for child in g.msgs(5):
                    name, ctok = child.str(1), child.int(3, signed32=True)
                    if name == "flows":
                        st["workflows"] += ctok
                    elif name == "skills":
                        st["skills"] += ctok
                    elif name == "subagents":
                        st["subagents"] += ctok
                    else:
                        continue
                    extracted += ctok
                st["system_prompt"] = gtok - extracted
                seen_prompt = True
            elif gtype == TOKEN_TOOLS:
                st["system_tools"] = gtok
                seen_tools = True
            elif gtype == TOKEN_RULES:
                st["rules"] += gtok
            elif gtype == TOKEN_MCP_TOOLS:
                st["mcp_tools"] += gtok
            elif gtype == TOKEN_SKILLS:
                st["skills"] += gtok
            elif gtype == TOKEN_SUBAGENTS:
                st["subagents"] += gtok
            elif gtype == TOKEN_CHAT_MESSAGES:
                chat_tok, chat_items = gtok, g.msgs(5)
            elif gtype == TOKEN_WORKFLOWS:
                st["legacy_workflows"] += gtok
            elif gtype == TOKEN_PLUGINS:
                st["plugins"] += gtok
            elif gtype == TOKEN_FLOWS:
                st["workflows"] += gtok

        if not seen_prompt and not seen_tools and chat_tok:
            usage = chat.msg(4) if chat is not None else None
            api_input = usage.int(2) + usage.int(5) if usage is not None else 0
            if chat_tok < api_input:
                st["system_overhead"] = api_input - chat_tok

        if chat_tok and not has_newer:
            ok, cats = chat_tokens_by_category(chat_items)
            if ok:
                st["user"], st["agent"], st["tool"], st["system_messages"] = cats
            else:
                est = st["user"] + st["agent"] + st["tool"]
                if est == 0:
                    st["agent"] = chat_tok
                else:
                    f = chat_tok / est
                    st["user"] = int(st["user"] * f)
                    st["agent"] = int(st["agent"] * f)
                    st["tool"] = int(st["tool"] * f)

    if chat_start is not None:
        st["active_checkpoint"] = chat_start.int(2, signed32=True)
    if has_newer and newer_step is not None:
        st["agent"] += checkpoint_summary_tokens(newer_step)
        newer_ck = newer_step.payload(STEP_CHECKPOINT)
        if newer_ck is not None:
            st["active_checkpoint"] = newer_ck.int(1)

    st["total"] = sum(
        st[k]
        for k in (
            "user", "agent", "tool", "system_messages", "system_prompt", "system_tools", "system_overhead",
            "rules", "skills", "subagents", "legacy_workflows", "workflows", "plugins", "mcp_tools",
        )
    )
    st["limit"] = limit
    st["checkpoints"] = checkpoints
    return st


def stats_heading(st: dict[str, Any]) -> str:
    if st["compacted"]:
        return "Estimated after compaction · updates next turn"
    if st["has_breakdown"]:
        return "Token usage by category"
    return "Estimated usage (awaiting generation)"


def checkpoint_view(st: dict[str, Any]) -> dict[str, Any] | None:
    cks = st["checkpoints"]
    if not cks:
        return None
    active = None
    historical = 0
    for ck in cks:
        if st["active_checkpoint"] >= 0 and ck["index"] == st["active_checkpoint"]:
            active = ck
        else:
            historical += 1
    if active is None:
        active, historical = cks[-1], len(cks) - 1
    text = (active["intent"] or "").strip()
    if len(text.encode()) > 70:
        text = text.encode()[:67].decode("utf-8", errors="ignore") + "..."
    if not text:
        text = f"steps {active['start']}–{active['end']}"
    return {
        "count": len(cks),
        "active_number": active["index"] + 1,
        "active_text": text,
        "historical": historical,
    }


def model_id(gen: Msg | None) -> str:
    planner = gen.msg(3, _PLANNER_FIELDS) if gen is not None else None
    return planner.str(28) if planner is not None else ""


def model_label(slug: str) -> str:
    parts = slug.split("-")
    effort = parts.pop().capitalize() if len(parts) > 1 and parts[-1] in EFFORT_LEVELS else ""
    words: list[str] = []
    for p in parts:
        if p.isdigit() and words and words[-1][-1:].isdigit():
            words[-1] += "." + p
        elif p in WORD_LABELS:
            words.append(WORD_LABELS[p])
        elif p[:1].isdigit():
            words.append(p.upper())
        else:
            words.append(p.capitalize())
    label = " ".join(words).replace("GPT OSS", "GPT-OSS")
    return f"{label} ({effort})" if effort else label


def model_name(gen: Msg | None, slug_fallback: str = "") -> str:
    chat = _gen_chat(gen)
    if chat is not None and chat.str(21):
        return chat.str(21)
    slug = model_id(gen) or slug_fallback
    if slug:
        return model_label(slug)
    if chat is not None and chat.str(19):
        return chat.str(19)
    return "Unknown Model"


def request_budget(gen: Msg | None) -> int:
    chat = _gen_chat(gen)
    start = chat.msg(9) if chat is not None else None
    cwm = start.msg(10) if start is not None else None
    return cwm.int(4, signed32=True) if cwm is not None else 0


def builtin_window_limit(slug: str) -> int:
    for prefix, tokens in NATIVE_CONTEXT_WINDOWS:
        if slug.startswith(prefix):
            return tokens
    return 0


def load_model_catalog(app_dir: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    env_cat = os.environ.get("ANTIGRAVITY_MODEL_CATALOG")
    if env_cat:
        candidates.append(Path(env_cat).expanduser())
    candidates.append(app_dir / "sidecar_data" / "desktop-context" / "context" / "data" / "model-catalog.json")
    for p in candidates:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {}


def parse_trajectory_lineage(tmb: Msg | None) -> dict[str, Any]:
    if tmb is None:
        return {
            "parent_conversation_id": "",
            "nesting_depth": 0,
            "role": "",
            "type_name": "",
            "model_enum": 0,
        }
    parent_id = tmb.str(5)
    spec = tmb.msg(8)
    return {
        "parent_conversation_id": parent_id,
        "nesting_depth": tmb.int(17) or (1 if parent_id else 0),
        "role": spec.str(2) if spec is not None else "",
        "type_name": spec.str(1) if spec is not None else "",
        "model_enum": spec.int(7) if spec is not None else 0,
    }


def extract_spawned_subagents(steps: list[Step]) -> dict[str, dict[str, str]]:
    """Extract {sub_cid: {role, type_name}} from invoke_subagent steps in parent .db."""
    out: dict[str, dict[str, str]] = {}
    for step in steps:
        tc = step.md.msg(4)
        tool = tc.str(2) if tc is not None else ""
        if tool != "invoke_subagent" and step.type != 127:
            continue
        if step.generic_pb is None:
            continue
        res = step.generic_pb.msg(2)
        out_text = res.str(1) if res is not None else ""
        m = SUBAGENT_CID_RE.search(out_text)
        if not m:
            continue
        sub_cid = m.group(1).lower()
        args: dict[str, Any] = {}
        if tc is not None and tc.str(3):
            try:
                parsed = json.loads(tc.str(3))
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                pass
        if not args:
            args = {kv.str(1): kv.str(2).strip('"') for kv in step.generic_pb.msgs(1) if kv.str(1)}
        role = str(args.get("Role") or "").strip().strip('"')
        tname = str(args.get("TypeName") or "").strip().strip('"')
        if sub_cid not in out:
            out[sub_cid] = {"role": role, "type_name": tname}
    return out


def _resolve_run_state(
    steps: list[Step],
    term_reason_code: int,
    term_step_idx: int,
    is_subagent: bool,
    summary_status: str = "",
    killed: bool = False,
    step_count: int | None = None,
    has_reported_override: bool | None = None,
) -> tuple[str, str]:
    term_reason = TERM_REASON_NAMES.get(term_reason_code, "")
    if killed:
        return "killed", term_reason

    if has_reported_override is not None:
        has_reported = has_reported_override
    else:
        has_reported = False
        if is_subagent:
            for s in steps:
                if s.status != STATUS_DONE:
                    continue
                tc = s.md.msg(4)
                if tc is not None and tc.str(2) == "send_message":
                    has_reported = True
                    break

    total_steps = len(steps) if step_count is None else step_count
    last_step = steps[-1] if steps else None
    executor_active = bool(total_steps and (term_step_idx < 0 or term_step_idx < total_steps - 1))

    if last_step is not None and last_step.status == STATUS_GENERATING:
        state = "generating"
    elif last_step is not None and last_step.status in (STATUS_PENDING, STATUS_RUNNING, STATUS_WAITING):
        state = "running"
    elif last_step is not None and last_step.status in (STATUS_ERROR, STATUS_INVALID):
        state = "error"
    elif last_step is not None and last_step.status in (STATUS_CANCELED, STATUS_INTERRUPTED, STATUS_HALTED):
        state = "canceled"
    elif term_reason_code == 1:
        state = "error"
    elif has_reported:
        state = "reported"
    elif term_reason_code == 2:
        state = "canceled"
    elif executor_active and summary_status != "CASCADE_RUN_STATUS_IDLE":
        state = "running"
    elif not total_steps:
        state = "starting"
    else:
        state = "idle"

    if summary_status == "CASCADE_RUN_STATUS_RUNNING" and state not in ("generating", "running"):
        state = "running"
    elif summary_status == "CASCADE_RUN_STATUS_IDLE" and state in ("running", "generating"):
        state = "reported" if has_reported else "idle"

    return state, term_reason


def _has_category_breakdown(gen: Msg | None) -> bool:
    chat = _gen_chat(gen)
    chat_start = chat.msg(9) if chat is not None else None
    cwm = chat_start.msg(10) if chat_start is not None else None
    breakdown = cwm.msg(3) if cwm is not None else None
    if breakdown is None:
        return False
    for g in breakdown.msgs(1):
        if g.int(2) == TOKEN_CHAT_MESSAGES and g.int(4, signed32=True):
            ok, _ = chat_tokens_by_category(g.msgs(5))
            return ok
    return False


def inspect_conversation_db(
    cid: str,
    app_dir: Path,
    catalog: dict[str, Any] | None = None,
    summary_meta: dict[str, Any] | None = None,
    include_spawned: bool = True,
) -> dict[str, Any] | None:
    """Read conversations/<cid>.db read-only and return structured context metrics, or None if .db missing."""
    db_path = app_dir / "conversations" / f"{cid}.db"
    if not db_path.is_file():
        return None

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    tmb = None
    term_reason_code = 0
    term_step_idx = -1
    try:
        gen_row = conn.execute("SELECT data FROM gen_metadata ORDER BY idx DESC LIMIT 1").fetchone()
        gen = Msg(gen_row[0], _GEN_FIELDS) if (gen_row and gen_row[0]) else None
        try:
            trow = conn.execute("SELECT data FROM trajectory_metadata_blob LIMIT 1").fetchone()
            if trow and trow[0]:
                tmb = Msg(trow[0])
        except sqlite3.Error:
            pass
        try:
            erow = conn.execute("SELECT data FROM executor_metadata ORDER BY idx DESC LIMIT 1").fetchone()
            if erow and erow[0]:
                em = Msg(erow[0])
                term_reason_code = em.int(1)
                term_step_idx = em.int(3)
        except sqlite3.Error:
            pass

        lineage = parse_trajectory_lineage(tmb)
        cat_map = catalog if catalog is not None else load_model_catalog(app_dir)

        slug = model_id(gen) or ENUM_MODEL_IDS.get(lineage["model_enum"], "")
        cat_entry = cat_map.get(slug) if slug else None
        cat_entry = cat_entry if isinstance(cat_entry, dict) else {}

        cat_max = int(cat_entry.get("maxTokens") or 0)
        raw_req_budget = request_budget(gen)
        if cat_max > 0:
            limit = cat_max
            limit_source = "catalog"
        else:
            b_win = builtin_window_limit(slug)
            if b_win > 0:
                limit = b_win
                limit_source = "builtin"
            elif raw_req_budget > 0:
                limit = raw_req_budget
                limit_source = "request_budget"
            else:
                limit = 0
                limit_source = "unknown"

        model_disp = str(cat_entry.get("displayName") or "").strip() or model_name(gen, slug)
        smeta = summary_meta or {}
        is_subagent = bool(lineage["parent_conversation_id"] or smeta.get("parent_conversation_id"))

        step_count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        last_row = (
            conn.execute(
                "SELECT idx, step_type, status, metadata, step_payload FROM steps ORDER BY idx DESC LIMIT 1"
            ).fetchone()
            if step_count > 0
            else None
        )

        use_fast = (step_count == 0) or (
            _has_category_breakdown(gen) and last_row is not None and last_row[3] is not None
        )
        st: dict[str, Any] | None = None
        budget_used = 0
        state_steps: list[Step] = []
        has_reported_override: bool | None = None
        spawned: dict[str, dict[str, str]] = {}

        if use_fast:
            if step_count == 0:
                st = collect_stats([], gen, limit)
                budget_used = int(st["total"])
                has_reported_override = False
            else:
                ckpt_rows = conn.execute(
                    "SELECT idx, step_type, status, metadata, step_payload FROM steps "
                    "WHERE step_type = 23 ORDER BY idx"
                ).fetchall()
                ckpt_steps = [Step(pos, r[4] or b"", r[1], r[2], r[3]) for pos, r in enumerate(ckpt_rows)]
                last_done_ckpt_sql_idx = -1
                last_done_ckpt_pos = -1
                last_done_ckpt_step: Step | None = None
                for pos, (r, s) in enumerate(zip(ckpt_rows, ckpt_steps)):
                    if s.payload(STEP_CHECKPOINT) is not None and s.status == STATUS_DONE:
                        last_done_ckpt_sql_idx = r[0]
                        last_done_ckpt_pos = pos
                        last_done_ckpt_step = s

                has_newer = False
                if last_done_ckpt_step is not None:
                    has_planner_after = (
                        conn.execute(
                            "SELECT 1 FROM steps WHERE idx > ? AND step_type = 15 AND status = ? LIMIT 1",
                            (last_done_ckpt_sql_idx, STATUS_DONE),
                        ).fetchone()
                        is not None
                    )
                    if not has_planner_after:
                        chat = _gen_chat(gen)
                        chat_start = chat.msg(9) if chat is not None else None
                        gen_ck = chat_start.int(2, signed32=True) if chat_start is not None else -1
                        ck_pb = last_done_ckpt_step.payload(STEP_CHECKPOINT)
                        ck_idx = ck_pb.int(1) if ck_pb is not None else 0
                        has_newer = gen_ck < ck_idx

                if has_newer:
                    post_rows = conn.execute(
                        "SELECT idx, step_type, status, metadata, step_payload FROM steps "
                        "WHERE idx > ? AND step_type != 23 ORDER BY idx",
                        (last_done_ckpt_sql_idx,),
                    ).fetchall()
                    post_steps = [Step(pos, r[4] or b"", r[1], r[2], r[3]) for pos, r in enumerate(post_rows)]
                    st = collect_stats(
                        ckpt_steps + post_steps,
                        gen,
                        limit,
                        newer_override=(last_done_ckpt_pos, last_done_ckpt_step),
                    )
                    budget_used = int(st["total"])
                else:
                    st = collect_stats(ckpt_steps, gen, limit, newer_override=(-1, None))
                    pos_planner_sql_idx = -1
                    prefix_tok = 0
                    for r_idx, r_meta in conn.execute(
                        "SELECT idx, metadata FROM steps WHERE step_type = 15 AND idx > ? ORDER BY idx DESC",
                        (last_done_ckpt_sql_idx,),
                    ):
                        if not r_meta:
                            continue
                        u = Msg(r_meta, _MD_USAGE_FIELDS).msg(9)
                        if u is None:
                            continue
                        inp, out_tok, cache_r, cache_w = u.int(2), u.int(3), u.int(5), u.int(6)
                        if inp + cache_r + cache_w > 0:
                            pos_planner_sql_idx = r_idx
                            prefix_tok = inp + cache_r + cache_w + out_tok
                            break
                    if pos_planner_sql_idx >= 0:
                        tail_rows = conn.execute(
                            "SELECT idx, step_type, status, metadata, step_payload FROM steps "
                            "WHERE idx > ? AND step_type != 23 ORDER BY idx",
                            (pos_planner_sql_idx,),
                        ).fetchall()
                        tail_steps = [Step(pos, r[4] or b"", r[1], r[2], r[3]) for pos, r in enumerate(tail_rows)]
                        delta_tok = sum(
                            post_turn_step_tokens(s) for s in tail_steps if s.payload(STEP_CHECKPOINT) is None
                        )
                        budget_used = max(int(st["total"]), prefix_tok + delta_tok)
                    else:
                        budget_used = int(st["total"])

                state_steps = [
                    Step(step_count - 1, last_row[4] or b"", last_row[1], last_row[2], last_row[3])
                ]
                has_reported_override = False
                if is_subagent:
                    for (r_meta,) in conn.execute(
                        "SELECT metadata FROM steps WHERE status = ? AND instr(metadata, ?) > 0",
                        (STATUS_DONE, b"send_message"),
                    ):
                        tc = Msg(r_meta, _MD_TC_FIELDS).msg(4)
                        if tc is not None and tc.str(2) == "send_message":
                            has_reported_override = True
                            break
                if include_spawned:
                    inv_rows = conn.execute(
                        "SELECT idx, step_type, status, metadata, step_payload FROM steps "
                        "WHERE step_type = 127 OR instr(metadata, ?) > 0 ORDER BY idx",
                        (b"invoke_subagent",),
                    ).fetchall()
                    inv_steps = [Step(pos, r[4] or b"", r[1], r[2], r[3]) for pos, r in enumerate(inv_rows)]
                    spawned = extract_spawned_subagents(inv_steps)
        else:
            rows = conn.execute(
                "SELECT idx, step_type, status, metadata, step_payload FROM steps ORDER BY idx"
            ).fetchall()
            steps = [Step(pos, r[4] or b"", r[1], r[2], r[3]) for pos, r in enumerate(rows)]
            step_count = len(steps)
            st = collect_stats(steps, gen, limit)
            budget_used = effective_budget_tokens(steps, st)
            state_steps = steps
            has_reported_override = None
            spawned = extract_spawned_subagents(steps) if include_spawned else {}
    finally:
        conn.close()

    if raw_req_budget > 0:
        req_budget = raw_req_budget
        budget_estimated = False
    elif limit > 256_000:
        # shortcut: pre-generation .db lacks cwm.max_context_tokens, updates on first completed turn
        req_budget = 256_000
        budget_estimated = True
    else:
        req_budget = limit
        budget_estimated = bool(limit > 0)

    budget_pct = round(pct(budget_used, req_budget), 1) if req_budget else 0.0
    used_pct = round(min(100.0, pct(st["total"], limit)), 1) if limit else 0.0

    state, term_reason = _resolve_run_state(
        state_steps,
        term_reason_code,
        term_step_idx,
        is_subagent=is_subagent,
        summary_status=str(smeta.get("status") or ""),
        killed=bool(smeta.get("killed")),
        step_count=step_count,
        has_reported_override=has_reported_override,
    )

    sys_total = sum(
        st[k]
        for k in (
            "system_prompt", "system_tools", "system_overhead", "rules", "skills",
            "subagents", "legacy_workflows", "workflows", "plugins", "mcp_tools", "system_messages",
        )
    )

    return {
        "conversation_id": cid,
        "parent_conversation_id": lineage["parent_conversation_id"] or str(smeta.get("parent_conversation_id") or ""),
        "nesting_depth": lineage["nesting_depth"] or int(smeta.get("nesting_depth") or 0),
        "role": lineage["role"] or str(smeta.get("role") or ""),
        "type_name": lineage["type_name"] or str(smeta.get("type_name") or smeta.get("agent_name") or ""),
        "model": model_disp,
        "state": state,
        "termination_reason": term_reason,
        "step_count": step_count,
        "heading": stats_heading(st),
        "used_tokens": st["total"],
        "limit_tokens": limit,
        "limit_source": limit_source,
        "used_pct": used_pct,
        "free_tokens": max(0, limit - st["total"]) if limit else 0,
        "request_budget_tokens": req_budget,
        "budget_estimated": budget_estimated,
        "budget_used_tokens": budget_used,
        "budget_pct": budget_pct,
        "budget_remaining_tokens": max(0, req_budget - budget_used) if req_budget else 0,
        "compacted": st["compacted"],
        "checkpoint_count": len(st["checkpoints"]),
        "checkpoint_buffer": st["checkpoint"],
        "checkpoint_view": checkpoint_view(st),
        "categories": [
            {"name": name, "key": key, "tokens": st[key], "pct": round(pct(st[key], limit), 1)}
            for name, key, always in CATEGORIES
            if always or st[key]
        ],
        "split": {
            "user": st["user"],
            "agent": st["agent"],
            "tool": st["tool"],
            "system": sys_total,
        },
        "spawned_subagents": spawned,
    }


def format_compact_context_tag(ctx: dict[str, Any] | None) -> str:
    """One-line compact context tag for list_conversations and get_conversation_digest headers."""
    if ctx is None:
        return "ctx=unavailable (no .db)"
    parts: list[str] = []
    if ctx.get("state"):
        parts.append(f"state={ctx['state']}")
    used_s = fmt_tok(ctx["used_tokens"])
    req_b = ctx["request_budget_tokens"]
    if req_b > 0:
        est_mark = "~" if ctx.get("budget_estimated") else ""
        b_used_s = fmt_tok(ctx["budget_used_tokens"])
        parts.append(f"ctx={used_s}")
        parts.append(f"budget={b_used_s}/{est_mark}{fmt_tok(req_b)} ({ctx['budget_pct']:.1f}%)")
    elif ctx["limit_tokens"] > 0:
        parts.append(f"ctx={used_s}/{fmt_tok(ctx['limit_tokens'])} ({ctx['used_pct']:.1f}%)")
    else:
        parts.append(f"ctx={used_s}")
    if ctx["checkpoint_count"] > 0:
        parts.append(f"compacted={ctx['checkpoint_count']}")
    elif ctx["compacted"]:
        parts.append("compacted=pending")
    return " | ".join(parts)


def render_context_report(ctx: dict[str, Any], subagent_contexts: list[dict[str, Any]] | None = None) -> str:
    """Format structured context stats into a compact, agent-friendly markdown/text report."""
    lines = [f"Conversation: {ctx['conversation_id']}"]
    if ctx["parent_conversation_id"] or ctx["role"] or ctx["type_name"]:
        role_label = ctx["role"] or "(untitled)"
        t_part = f" ({ctx['type_name']})" if ctx["type_name"] else ""
        p_part = f" | parent={ctx['parent_conversation_id']}" if ctx["parent_conversation_id"] else ""
        d_part = f" | depth={ctx['nesting_depth']}" if ctx["nesting_depth"] else ""
        lines.append(f"Subagent: {role_label}{t_part}{p_part}{d_part}")

    term_part = (
        f" ({ctx['termination_reason']})"
        if ctx["termination_reason"] and ctx["termination_reason"] not in ("no_tool_call", "terminal_step_type")
        else ""
    )
    lines.append(f"State: {ctx['state']}{term_part} | Steps: {ctx['step_count']} | Model: {ctx['model']}")

    used = ctx["used_tokens"]
    limit = ctx["limit_tokens"]
    if limit > 0:
        src_note = ", builtin window" if ctx["limit_source"] == "builtin" else (
            ", from request budget" if ctx["limit_source"] == "request_budget" else ""
        )
        lines.append(
            f"Context window: {fmt_tok(used)} / {fmt_tok(limit)} tokens "
            f"({ctx['used_pct']:.1f}%, free {fmt_tok(ctx['free_tokens'])}{src_note})"
        )
    else:
        lines.append(f"Context window: {fmt_tok(used)} / unknown limit")

    req_b = ctx["request_budget_tokens"]
    if req_b > 0:
        est_note = ", estimated default budget" if ctx["budget_estimated"] else ""
        lines.append(
            f"Compaction budget: {fmt_tok(ctx['budget_used_tokens'])} / {fmt_tok(req_b)} tokens "
            f"({ctx['budget_pct']:.1f}%, remaining {fmt_tok(ctx['budget_remaining_tokens'])}{est_note})"
        )

    lines.append(f"Breakdown ({ctx['heading']}):")
    for c in ctx["categories"]:
        pct_part = f" ({c['pct']:.1f}%)" if limit > 0 else ""
        lines.append(f"  - {c['name']}: {fmt_tok(c['tokens'])} tokens{pct_part}")

    ck_view = ctx["checkpoint_view"]
    if ck_view is not None:
        hist_part = f"; {ck_view['historical']} historical summarized" if ck_view["historical"] else ""
        buf_part = (
            f"; buffer {fmt_tok(ctx['checkpoint_buffer'])} not counted in usage"
            if ctx["checkpoint_buffer"]
            else ""
        )
        lines.append(
            f"Checkpoints: {ck_view['count']} (active #{ck_view['active_number']}: "
            f"{ck_view['active_text']}{hist_part}{buf_part})"
        )
    else:
        lines.append("Checkpoints: 0 (no compaction)")

    if subagent_contexts is not None:
        if not subagent_contexts:
            lines.append("Subagents: 0")
        else:
            lines.append(f"Subagents ({len(subagent_contexts)}):")
            for sc in subagent_contexts:
                scid = sc["conversation_id"]
                if sc.get("_missing_db"):
                    r_lbl = sc.get("role") or "(untitled)"
                    lines.append(f"  - {scid} | {r_lbl} | ctx=unavailable (no .db)")
                    continue
                role_s = sc["role"] or sc["type_name"] or "(untitled)"
                type_s = f" ({sc['type_name']})" if sc["type_name"] and sc["role"] else ""
                sp = sc["split"]
                split_parts = [f"tools={fmt_tok(sp['tool'])}", f"agent={fmt_tok(sp['agent'])}"]
                if sp["user"] > 0:
                    split_parts.append(f"user={fmt_tok(sp['user'])}")
                split_parts.append(f"sys={fmt_tok(sp['system'])}")
                ck_s = f"compacted={sc['checkpoint_count']}"
                if sc["compacted"]:
                    ck_s += " (post-compaction est.)"
                req_sb = sc["request_budget_tokens"]
                est_mark = "~" if sc.get("budget_estimated") else ""
                b_str = (
                    f"{fmt_tok(sc['budget_used_tokens'])}/{est_mark}{fmt_tok(req_sb)} ({sc['budget_pct']:.1f}%)"
                    if req_sb > 0
                    else "n/a"
                )
                w_str = (
                    f"{fmt_tok(sc['used_tokens'])}/{fmt_tok(sc['limit_tokens'])} ({sc['used_pct']:.1f}%)"
                    if sc["limit_tokens"] > 0
                    else fmt_tok(sc["used_tokens"])
                )
                lines.append(
                    f"  - {scid} | [{sc['state'].upper()}] {role_s}{type_s} | {sc['model']} | steps={sc['step_count']}\n"
                    f"    ctx={w_str} | budget={b_str} | {', '.join(split_parts)} | {ck_s}"
                )

    return "\n".join(lines)
