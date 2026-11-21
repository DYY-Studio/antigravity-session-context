#!/usr/bin/env python3
"""Lightweight MCP server for Antigravity cross-conversation context lookup and digest."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
from urllib.parse import unquote, urlparse

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from context_stats import (
    format_compact_context_tag,
    inspect_conversation_db,
    load_model_catalog,
    render_context_report,
)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
USER_REQ_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.DOTALL)
MSG_HEADER_RE = re.compile(r"^\[Message\]\s+timestamp=\S+\s+sender=(\S+)\s+priority=\S+\s+content=(.*)", re.DOTALL)
SUBAGENT_CID_RE = re.compile(r'"conversationId"\s*:\s*"([0-9a-fA-F-]{36})"')
STEP_INDEX_PREFIX_RE = re.compile(r'^\s*\{\s*"step_index"\s*:\s*(\d+)')
TYPE_FIELD_RE = re.compile(r'"type"\s*:\s*"([A-Z_]+)"')
CREATED_AT_RE = re.compile(r'"created_at"\s*:\s*"([^"]*)"')

_CACHE_LIMIT = 512
_DB_CTX_CACHE: dict[tuple[Any, ...], dict[str, Any] | None] = {}
_METRICS_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
_SUBAGENT_MAP_CACHE: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
_CATALOG_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _cache_put(cache: dict[Any, Any], key: Any, val: Any, limit: int = _CACHE_LIMIT) -> None:
    if len(cache) >= limit and key not in cache:
        cache.pop(next(iter(cache)))
    cache[key] = val


def _file_stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _cached_model_catalog(app_dir: Path) -> dict[str, Any]:
    env_cat = os.environ.get("ANTIGRAVITY_MODEL_CATALOG")
    cand = Path(env_cat).expanduser() if env_cat else (
        app_dir / "sidecar_data" / "desktop-context" / "context" / "data" / "model-catalog.json"
    )
    st = _file_stat_key(cand)
    if st is not None:
        ckey = (str(cand), st[0], st[1])
        cached = _CATALOG_CACHE.get(ckey)
        if cached is not None:
            return cached
        data = load_model_catalog(app_dir)
        _cache_put(_CATALOG_CACHE, ckey, data, limit=16)
        return data
    return load_model_catalog(app_dir)


def _cached_inspect_db(
    cid: str,
    app_dir: Path,
    catalog: dict[str, Any] | None = None,
    summary_meta: dict[str, Any] | None = None,
    include_spawned: bool = True,
) -> dict[str, Any] | None:
    db_path = app_dir / "conversations" / f"{cid}.db"
    db_st = _file_stat_key(db_path)
    if db_st is None:
        return None
    wal_st = _file_stat_key(db_path.with_name(f"{cid}.db-wal"))
    smeta_key = tuple(sorted((k, v) for k, v in (summary_meta or {}).items()))
    ckey = (cid, include_spawned, db_st, wal_st, smeta_key)
    if ckey in _DB_CTX_CACHE:
        cached = _DB_CTX_CACHE[ckey]
        return dict(cached) if cached is not None else None
    res = inspect_conversation_db(
        cid,
        app_dir,
        catalog=catalog,
        summary_meta=summary_meta,
        include_spawned=include_spawned,
    )
    _cache_put(_DB_CTX_CACHE, ckey, res)
    return dict(res) if res is not None else None


def app_data_dir() -> Path:
    env_dir = os.environ.get("ANTIGRAVITY_APP_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    gemini_dir = os.environ.get("ANTIGRAVITY_GEMINI_DIR")
    if gemini_dir:
        return Path(gemini_dir).expanduser() / "antigravity"
    return Path.home() / ".gemini" / "antigravity"


def open_summaries_db() -> sqlite3.Connection:
    db_path = app_data_dir() / "conversation_summaries.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"Conversation summaries database not found: {db_path}")
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_arg(args: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in args and args[k] is not None:
            return args[k]
    return default


def validate_conversation_id(cid: str) -> str:
    cid = (cid or "").strip()
    if not UUID_RE.match(cid):
        raise ValueError(f"Invalid conversation_id (expected UUID): {cid!r}")
    return cid.lower()


def uri_to_path(uri: str) -> str:
    s = str(uri or "").strip()
    if s.startswith("file:"):
        parsed = urlparse(s)
        p = unquote(parsed.path)
        if len(p) > 2 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        if parsed.netloc and parsed.netloc != "localhost":
            p = f"//{parsed.netloc}{p}"
        return p
    return s


def parse_workspaces(raw_uris: str) -> list[str]:
    if not raw_uris:
        return []
    try:
        items = json.loads(raw_uris)
        if isinstance(items, list):
            return [uri_to_path(str(x)) for x in items]
    except json.JSONDecodeError:
        pass
    return [uri_to_path(raw_uris)]


def parse_iso_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    sec = max(0.0, seconds)
    whole = int(round(sec))
    if whole >= 60:
        mins = sec / 60.0
        return f"{whole}s ({mins:.1f}m)"
    return f"{whole}s"


def compact_ts(ts: str) -> str:
    s = str(ts or "").strip()
    dt = parse_iso_ts(s)
    if dt is not None:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return s


def extract_user_text(raw: str) -> str:
    m = USER_REQ_RE.search(raw or "")
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", raw or "", flags=re.DOTALL)
    cleaned = re.sub(r"<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def extract_system_message(raw: str) -> tuple[str, str]:
    """Return (sender, cleaned_text) from a SYSTEM_MESSAGE payload."""
    inner = (raw or "").strip()
    if "<SYSTEM_MESSAGE>" in inner:
        inner = inner.split("<SYSTEM_MESSAGE>")[-1]
    if inner.endswith("</SYSTEM_MESSAGE>"):
        inner = inner[: -len("</SYSTEM_MESSAGE>")]
    inner = inner.strip()
    mh = MSG_HEADER_RE.match(inner)
    if mh:
        sender = mh.group(1).strip()
        body = mh.group(2).strip()
        return sender, f"[from {sender}] {body}"
    return "", inner


def _unwrap_arg_value(val: Any) -> Any:
    if isinstance(val, str):
        s = val.strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "{" and s[-1] == "}") or (s[0] == "[" and s[-1] == "]")):
            try:
                return json.loads(s)
            except Exception:
                if s[0] == '"' and s[-1] == '"':
                    return s[1:-1]
    return val


def _clean_scalar(val: Any, max_len: int = 80, keep_tail: bool = False) -> str:
    unwrapped = _unwrap_arg_value(val)
    if isinstance(unwrapped, (dict, list)):
        text = json.dumps(unwrapped, ensure_ascii=False)
    else:
        text = str(unwrapped)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        if keep_tail and max_len >= 30:
            head_len = 22
            tail_len = max_len - head_len - 3
            return text[:head_len] + "..." + text[-tail_len:]
        return text[: max_len - 3] + "..."
    return text


def summarize_tool_call(tc: dict[str, Any]) -> str:
    name = tc.get("name", "unknown")
    args = tc.get("args") or {}
    if not isinstance(args, dict):
        return f"{name}()"

    if name == "call_mcp_tool":
        srv = _clean_scalar(args.get("ServerName", ""), 40)
        tname = _clean_scalar(args.get("ToolName", ""), 40)
        call_name = f"{srv}.{tname}" if (srv and tname) else (tname or srv or name)
        raw_inner = _unwrap_arg_value(args.get("Arguments"))
        if isinstance(raw_inner, dict) and raw_inner:
            inner_items = [f"{k}={_clean_scalar(v, 48)}" for k, v in raw_inner.items() if v is not None]
            inner_str = ", ".join(inner_items)
            if len(inner_str) > 120:
                inner_str = inner_str[:117] + "..."
            return f"{call_name}({inner_str})"
        elif raw_inner is not None and raw_inner != "":
            return f"{call_name}({_clean_scalar(raw_inner, 100)})"
        return f"{call_name}()"

    highlights = []
    priority_keys = (
        "Path",
        "CommandLine",
        "toolSummary",
        "query",
        "queries",
        "Role",
        "TypeName",
        "Model",
        "Recipient",
        "Message",
        "ServerName",
        "ToolName",
        "Action",
        "DurationSeconds",
        "CronExpression",
    )
    for k in priority_keys:
        if k in args and args[k] is not None:
            is_path = k == "Path"
            highlights.append(f"{k}={_clean_scalar(args[k], 85 if is_path else 80, keep_tail=is_path)}")
    if not highlights and args:
        first_k = next(iter(args))
        is_path = first_k == "Path"
        highlights.append(f"{first_k}={_clean_scalar(args[first_k], 85 if is_path else 80, keep_tail=is_path)}")
    return f"{name}({', '.join(highlights)})"


def _parse_subagent_invocation(tc: dict[str, Any], generic_content: str, invoke_step: int) -> tuple[str, dict[str, Any]] | None:
    if tc.get("name") != "invoke_subagent":
        return None
    m = SUBAGENT_CID_RE.search(generic_content or "")
    if not m:
        return None
    sub_cid = m.group(1).lower()
    targs = tc.get("args") or {}
    return sub_cid, {
        "role": _clean_scalar(targs.get("Role", ""), 80) if targs.get("Role") else "",
        "type": _clean_scalar(targs.get("TypeName", ""), 40) if targs.get("TypeName") else "",
        "model": _clean_scalar(targs.get("Model", ""), 40) if targs.get("Model") else "",
        "invoke_step": invoke_step,
    }


def extract_parent_subagent_map(parent_cid: str) -> dict[str, dict[str, Any]]:
    """Map subagent conversation_id -> {role, type, model, invoke_step} from parent transcript."""
    transcript_path = app_data_dir() / "brain" / parent_cid / ".system_generated" / "logs" / "transcript.jsonl"
    st = _file_stat_key(transcript_path)
    if st is None:
        return {}
    ckey = (parent_cid, st[0], st[1])
    cached = _SUBAGENT_MAP_CACHE.get(ckey)
    if cached is not None:
        return cached

    mapping: dict[str, dict[str, Any]] = {}
    pending_calls: list[dict[str, Any]] = []
    generic_offset = 0
    pending_step = 0
    try:
        with transcript_path.open(encoding="utf-8") as f:
            for line in f:
                if not pending_calls and '"invoke_subagent"' not in line:
                    continue
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                stype = obj.get("type")
                idx = obj.get("step_index", 0)
                if stype == "PLANNER_RESPONSE":
                    tcalls = obj.get("tool_calls") or []
                    if any(tc.get("name") == "invoke_subagent" for tc in tcalls):
                        pending_calls = tcalls
                        generic_offset = 0
                        pending_step = idx
                    else:
                        pending_calls = []
                        generic_offset = 0
                elif stype == "GENERIC" and pending_calls:
                    if generic_offset < len(pending_calls):
                        parsed = _parse_subagent_invocation(
                            pending_calls[generic_offset], obj.get("content") or "", pending_step
                        )
                        if parsed:
                            mapping[parsed[0]] = parsed[1]
                    generic_offset += 1
                    if generic_offset >= len(pending_calls):
                        pending_calls = []
                else:
                    pending_calls = []
                    generic_offset = 0
    except Exception:
        pass
    _cache_put(_SUBAGENT_MAP_CACHE, ckey, mapping)
    return mapping


def compute_transcript_metrics(cid: str) -> dict[str, Any]:
    """Compute step count, planner turns, tool call count, and wall-clock duration from transcript.jsonl."""
    transcript_path = app_data_dir() / "brain" / cid / ".system_generated" / "logs" / "transcript.jsonl"
    st = _file_stat_key(transcript_path)
    if st is None:
        return {}
    ckey = (cid, st[0], st[1])
    cached = _METRICS_CACHE.get(ckey)
    if cached is not None:
        return cached

    steps = 0
    turns = 0
    tools = 0
    first_ts = ""
    last_ts_head = ""
    try:
        with transcript_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                steps += 1
                head = line[:300]
                if '"created_at"' in head:
                    if not first_ts:
                        cm = CREATED_AT_RE.search(head)
                        if cm and cm.group(1):
                            first_ts = cm.group(1)
                    last_ts_head = head
                tm = TYPE_FIELD_RE.search(head)
                if tm is not None:
                    if tm.group(1) == "PLANNER_RESPONSE":
                        turns += 1
                        if '"tool_calls"' in line:
                            obj = json.loads(line)
                            tools += len(obj.get("tool_calls") or [])
                else:
                    obj = json.loads(line)
                    created = obj.get("created_at") or ""
                    if created:
                        if not first_ts:
                            first_ts = created
                        last_ts_head = f'"created_at":"{created}"'
                    if obj.get("type") == "PLANNER_RESPONSE":
                        turns += 1
                        tools += len(obj.get("tool_calls") or [])
    except Exception:
        return {}
    last_ts = ""
    if last_ts_head:
        cm = CREATED_AT_RE.search(last_ts_head)
        if cm:
            last_ts = cm.group(1)
    t0 = parse_iso_ts(first_ts)
    t1 = parse_iso_ts(last_ts)
    dur_sec = (t1 - t0).total_seconds() if (t0 and t1) else None
    res = {
        "steps": steps,
        "turns": turns,
        "tools": tools,
        "duration_sec": dur_sec,
        "duration": format_duration(dur_sec),
    }
    _cache_put(_METRICS_CACHE, ckey, res)
    return res


def format_subagent_role_tag(role_info: dict[str, Any] | None, ctx: dict[str, Any] | None = None) -> str:
    rinfo = role_info or {}
    role = rinfo.get("role") or (ctx.get("role") if ctx else "") or ""
    tname = rinfo.get("type") or (ctx.get("type_name") if ctx else "") or ""
    model = rinfo.get("model") or ""
    if ctx and ctx.get("model") and ctx["model"] != "Unknown Model" and model.lower() in (
        "",
        "inherit",
        "flash",
        "flash_lite",
        "pro",
    ):
        model = ctx["model"]
    meta = [x for x in (tname, model) if x]
    meta_str = f" ({', '.join(meta)})" if meta else ""
    if role:
        return f"{role}{meta_str}"
    return meta_str.strip()


def list_artifacts(brain_dir: Path) -> list[str]:
    if not brain_dir.is_dir():
        return []
    skip = {".system_generated", ".user_uploaded", "scratch", ".DS_Store"}
    out = []
    for entry in sorted(brain_dir.iterdir(), key=lambda p: p.name):
        if entry.name in skip or entry.name.startswith("."):
            continue
        if entry.is_file():
            out.append(str(entry))
    return out


def tool_list_conversations(args: dict[str, Any]) -> str:
    workspace = str(get_arg(args, "workspace", default="") or "").strip()
    limit = max(1, min(int(get_arg(args, "limit", default=15)), 100))
    include_subagents = bool(get_arg(args, "include_subagents", "includeSubagents", default=False))
    parent_id = str(get_arg(args, "parent_conversation_id", "parentConversationId", default="") or "").strip()

    with open_summaries_db() as conn:
        sub_counts = dict(
            conn.execute(
                "SELECT parent_conversation_id, COUNT(*) FROM conversation_summaries "
                "WHERE parent_conversation_id != '' GROUP BY parent_conversation_id"
            ).fetchall()
        )

        clauses = []
        params: list[Any] = []
        if parent_id:
            clauses.append("parent_conversation_id = ?")
            params.append(validate_conversation_id(parent_id))
        elif not include_subagents:
            clauses.append("parent_conversation_id = ''")

        if workspace:
            slash_ws = workspace.replace("\\", "/")
            clauses.append("workspace_uris LIKE ?")
            params.append(f"%{slash_ws}%")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT conversation_id, title, step_count, last_modified_time, "
            "workspace_uris, parent_conversation_id, status, killed, agent_name, nesting_depth "
            f"FROM conversation_summaries {where_sql} "
            "ORDER BY last_modified_time DESC LIMIT ?"
        )
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return "No matching conversations found."

    app_dir = app_data_dir()
    catalog: dict[str, Any] | None = None
    parent_role_cache: dict[str, dict[str, dict[str, Any]]] = {}
    lines = [f"Found {len(rows)} conversation(s):"]
    for r in rows:
        cid = r["conversation_id"]
        parent_cid = r["parent_conversation_id"] or ""
        title = r["title"] or ""
        rinfo: dict[str, Any] | None = None
        sub_ctx: dict[str, Any] | None = None
        if parent_cid:
            if catalog is None:
                catalog = _cached_model_catalog(app_dir)
            if not title:
                if parent_cid not in parent_role_cache:
                    parent_role_cache[parent_cid] = extract_parent_subagent_map(parent_cid)
                rinfo = parent_role_cache[parent_cid].get(cid.lower())
            sub_ctx = _cached_inspect_db(
                cid,
                app_dir,
                catalog=catalog,
                summary_meta={
                    "status": r["status"],
                    "killed": r["killed"],
                    "agent_name": r["agent_name"],
                    "nesting_depth": r["nesting_depth"],
                    "parent_conversation_id": parent_cid,
                    "role": (rinfo or {}).get("role", ""),
                    "type_name": (rinfo or {}).get("type", ""),
                },
                include_spawned=False,
            )
            if not title:
                role_label = format_subagent_role_tag(rinfo, sub_ctx)
                title = f"[Subagent] {role_label}" if role_label else "(untitled subagent)"
        elif not title:
            title = "(untitled)"

        ws = ", ".join(parse_workspaces(r["workspace_uris"])) or "-"
        sub_n = sub_counts.get(cid, 0)
        sub_tag = f" | subagents={sub_n}" if sub_n else ""
        parent_tag = f" | parent={parent_cid}" if parent_cid else ""

        metrics_tag = f"steps={r['step_count']}"
        if parent_cid:
            m = compute_transcript_metrics(cid)
            if m:
                dur_part = f" | duration={m['duration']}" if m.get("duration") else ""
                metrics_tag = f"steps={m['steps']} | turns={m['turns']} | tools={m['tools']}{dur_part}"
            metrics_tag = f"{metrics_tag} | {format_compact_context_tag(sub_ctx)}"

        lines.append(
            f"- {cid} | {compact_ts(r['last_modified_time'])} | {metrics_tag}{sub_tag}{parent_tag}\n"
            f"  title: {title}\n"
            f"  workspace: {ws}"
        )
    return "\n".join(lines)


def build_fts_query(raw_query: str) -> str:
    tokens = [t for t in re.split(r"[\s|]+", raw_query.strip()) if t]
    if not tokens:
        raise ValueError("Search query must not be empty.")
    escaped = ['"' + t.replace('"', '""') + '"' for t in tokens]
    if "|" in raw_query:
        return " OR ".join(escaped)
    return " ".join(escaped)


def _parse_step_query(raw_query: str) -> tuple[bool, list[str]]:
    q = (raw_query or "").strip().lower()
    if not q:
        return False, []
    if "|" in q:
        return True, [b.strip() for b in q.split("|") if b.strip()]
    return False, [t for t in q.split() if t]


def match_query_in_step(
    obj: dict[str, Any],
    raw_query: str,
    parsed: tuple[bool, list[str]] | None = None,
) -> tuple[bool, bool]:
    """Return (matched, matched_in_thinking) for a step object against raw_query."""
    is_or, terms = parsed if parsed is not None else _parse_step_query(raw_query)
    if not terms:
        return True, False
    content_hay = (obj.get("content") or "").lower()
    thinking_hay = (obj.get("thinking") or "").lower()
    tcalls = obj.get("tool_calls")
    tools_hay = json.dumps(tcalls, ensure_ascii=False).lower() if tcalls else ""
    combined = f"{content_hay} {thinking_hay} {tools_hay}"

    if is_or:
        matched = any(b in combined for b in terms)
        in_thinking = any(b in thinking_hay for b in terms)
        return matched, in_thinking

    matched = all(t in combined for t in terms)
    in_thinking = any(t in thinking_hay for t in terms)
    return matched, in_thinking


def find_matching_steps(cid: str, raw_query: str, max_shown: int = 12) -> str:
    raw_q = (raw_query or "").strip()
    if not raw_q:
        return ""
    logs_dir = app_data_dir() / "brain" / cid / ".system_generated" / "logs"
    full_path = logs_dir / "transcript_full.jsonl"
    transcript_path = logs_dir / "transcript.jsonl"
    chosen_path = full_path if full_path.is_file() else transcript_path
    if not chosen_path.is_file():
        return ""

    tokens = [t.lower() for t in re.split(r"[\s|]+", raw_q) if t]
    is_or = "|" in raw_q
    can_raw_filter = all(not any(c in t for c in ("\\", '"', "\n", "\r", "\t")) for t in tokens)
    all_ascii = all(t.isascii() for t in tokens)
    all_matches: list[int] = []
    any_matches: list[int] = []
    try:
        with chosen_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                tm = TYPE_FIELD_RE.search(line[:260])
                if tm is not None and tm.group(1) not in ("USER_INPUT", "PLANNER_RESPONSE", "SYSTEM_MESSAGE"):
                    continue
                if can_raw_filter and (all_ascii or "\\u" not in line):
                    low_raw = line.lower()
                    if not any(t in low_raw for t in tokens):
                        continue
                obj = json.loads(line)
                stype = obj.get("type")
                if stype not in ("USER_INPUT", "PLANNER_RESPONSE", "SYSTEM_MESSAGE"):
                    continue
                idx = obj.get("step_index")
                if idx is None:
                    continue
                tcalls = obj.get("tool_calls")
                tools_str = json.dumps(tcalls, ensure_ascii=False) if tcalls else ""
                hay = ((obj.get("content") or "") + " " + (obj.get("thinking") or "") + " " + tools_str).lower()
                if not is_or and all(t in hay for t in tokens):
                    all_matches.append(idx)
                elif any(t in hay for t in tokens):
                    any_matches.append(idx)
    except Exception:
        return ""
    chosen = all_matches if (all_matches and not is_or) else any_matches
    if not chosen:
        return ""
    shown = ", ".join(str(i) for i in chosen[:max_shown])
    if len(chosen) > max_shown:
        return f"[{shown}, ... ({len(chosen)} steps)]"
    return f"[{shown}]"


def tool_search_conversations(args: dict[str, Any]) -> str:
    raw_query = str(get_arg(args, "query", default="") or "").strip()
    workspace = str(get_arg(args, "workspace", default="") or "").strip()
    parent_id = str(get_arg(args, "parent_conversation_id", "parentConversationId", default="") or "").strip()
    limit = max(1, min(int(get_arg(args, "limit", default=10)), 50))
    fts_query = build_fts_query(raw_query)

    with open_summaries_db() as conn:
        clauses = ["conversations_fts MATCH ?"]
        params: list[Any] = [fts_query]
        if workspace:
            slash_ws = workspace.replace("\\", "/")
            clauses.append("s.workspace_uris LIKE ?")
            params.append(f"%{slash_ws}%")
        if parent_id:
            clauses.append("s.parent_conversation_id = ?")
            params.append(validate_conversation_id(parent_id))

        sql = (
            "SELECT f.conversation_id, s.title, s.step_count, s.last_modified_time, "
            "s.workspace_uris, s.parent_conversation_id, "
            "snippet(conversations_fts, 13, '[', ']', '...', 24) AS user_snip, "
            "snippet(conversations_fts, 14, '[', ']', '...', 24) AS agent_snip "
            "FROM conversations_fts f "
            "LEFT JOIN conversation_summaries s ON s.conversation_id = f.conversation_id "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY bm25(conversations_fts) LIMIT ?"
        )
        params.append(limit * 3)
        rows = conn.execute(sql, params).fetchall()

        # Fallback: when searching within a specific parent_conversation_id, FTS may not index thinking/tool_calls
        fallback_rows = []
        if parent_id and len(rows) < limit:
            fallback_rows = conn.execute(
                "SELECT conversation_id, title, step_count, last_modified_time, workspace_uris, parent_conversation_id "
                "FROM conversation_summaries WHERE parent_conversation_id = ? ORDER BY last_modified_time DESC",
                (validate_conversation_id(parent_id),),
            ).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    for r in rows:
        cid = r["conversation_id"]
        if cid not in grouped:
            if len(grouped) >= limit:
                continue
            grouped[cid] = {
                "meta": r,
                "snippets": [],
            }
        for label, snip in (("user", r["user_snip"]), ("agent", r["agent_snip"])):
            cleaned = re.sub(r"\s+", " ", (snip or "").strip())
            if cleaned and "[" in cleaned and "]" in cleaned:
                entry = f"[{label}] {cleaned}"
                if entry not in grouped[cid]["snippets"] and len(grouped[cid]["snippets"]) < 3:
                    grouped[cid]["snippets"].append(entry)

    for fr in fallback_rows:
        cid = fr["conversation_id"]
        if cid in grouped or len(grouped) >= limit:
            continue
        hits = find_matching_steps(cid, raw_query)
        if hits:
            grouped[cid] = {"meta": fr, "snippets": [], "precomputed_hits": hits}

    if not grouped:
        return f"No conversations matched query: {raw_query!r}"

    parent_role_cache: dict[str, dict[str, dict[str, Any]]] = {}
    lines = [f"Matched {len(grouped)} conversation(s) for {raw_query!r}:"]
    for cid, info in grouped.items():
        r = info["meta"]
        parent_cid = r["parent_conversation_id"] or ""
        title = r["title"] or ""
        if not title and parent_cid:
            if parent_cid not in parent_role_cache:
                parent_role_cache[parent_cid] = extract_parent_subagent_map(parent_cid)
            role_label = format_subagent_role_tag(parent_role_cache[parent_cid].get(cid.lower()))
            title = f"[Subagent] {role_label}" if role_label else "(untitled subagent)"
        elif not title:
            title = "(untitled)"

        ws = ", ".join(parse_workspaces(r["workspace_uris"] or "")) or "-"
        parent_tag = f" | parent={parent_cid}" if parent_cid else ""
        step_hits = info.get("precomputed_hits") or find_matching_steps(cid, raw_query)
        step_hits_line = f"\n  matched_steps: {step_hits}" if step_hits else ""
        lines.append(
            f"- {cid} | {compact_ts(r['last_modified_time'])} | steps={r['step_count']}{parent_tag}\n"
            f"  title: {title}\n"
            f"  workspace: {ws}{step_hits_line}"
        )
        for s in info["snippets"]:
            lines.append(f"  match: {s}")
    return "\n".join(lines)


def load_full_steps(full_path: Path, step_indices: set[int]) -> dict[int, dict[str, Any]]:
    if not step_indices or not full_path.is_file():
        return {}
    found: dict[int, dict[str, Any]] = {}
    with full_path.open(encoding="utf-8") as f:
        for line in f:
            m = STEP_INDEX_PREFIX_RE.match(line)
            if m is not None:
                if int(m.group(1)) not in step_indices:
                    continue
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("step_index")
            if idx in step_indices:
                found[idx] = obj
                if len(found) == len(step_indices):
                    break
    return found


def _render_digest_step(
    obj: dict[str, Any],
    *,
    include_tools: bool,
    include_thinking: bool,
    include_system: bool,
    has_query: bool,
    raw_query: str,
    parsed_query: tuple[bool, list[str]] | None,
    planner_result_steps: dict[int, list[int]],
    result_step_subagent: dict[int, str],
    ask_question_pending: bool,
) -> tuple[str | None, bool]:
    """Render one digest step and return (rendered_block_or_None, next_ask_question_pending)."""
    idx = obj.get("step_index", 0)
    stype = obj.get("type")
    created = obj.get("created_at", "")

    if stype == "USER_INPUT":
        if has_query and not match_query_in_step(obj, raw_query, parsed_query)[0]:
            return None, False
        text = extract_user_text(obj.get("content", ""))
        return f"\n[Step {idx} | USER | {created}]\n{text}", False

    if stype == "PLANNER_RESPONSE":
        tcalls = obj.get("tool_calls") or []
        next_ask = any(tc.get("name") == "ask_question" for tc in tcalls)
        matched_query, matched_in_thinking = (
            match_query_in_step(obj, raw_query, parsed_query) if has_query else (True, False)
        )
        if not matched_query:
            return None, next_ask
        parts = []
        if (include_thinking or matched_in_thinking) and obj.get("thinking"):
            parts.append(f"<thinking>\n{obj['thinking'].strip()}\n</thinking>")
        if include_tools and tcalls:
            res_indices = planner_result_steps.get(idx, [])
            summaries = []
            for j, tc in enumerate(tcalls):
                sc = summarize_tool_call(tc)
                if j < len(res_indices):
                    r_idx = res_indices[j]
                    spawned_cid = result_step_subagent.get(r_idx)
                    if spawned_cid:
                        summaries.append(f"[Step {r_idx} -> {spawned_cid}] {sc}")
                    else:
                        summaries.append(f"[Step {r_idx}] {sc}")
                else:
                    summaries.append(sc)
            parts.append("Tools: " + "; ".join(summaries))
        content = (obj.get("content") or "").strip()
        if content:
            parts.append(content)
        if parts:
            return f"\n[Step {idx} | AGENT | {created}]\n" + "\n".join(parts), next_ask
        return None, next_ask

    if stype == "GENERIC" and ask_question_pending:
        if has_query and not match_query_in_step(obj, raw_query, parsed_query)[0]:
            return None, False
        ans = (obj.get("content") or "").strip()
        if ans:
            return f"\n[Step {idx} | USER_ANSWER | {created}]\n{ans}", False
        return None, False

    if stype == "SYSTEM_MESSAGE" and include_system:
        if has_query and not match_query_in_step(obj, raw_query, parsed_query)[0]:
            return None, ask_question_pending
        sender, clean_sys = extract_system_message(obj.get("content") or "")
        if not has_query and ("/task-" in sender or clean_sys.startswith("Task id ")):
            return None, ask_question_pending
        cap = 1200 if has_query else 400
        preview = clean_sys[:cap] + ("..." if len(clean_sys) > cap else "")
        return f"\n[Step {idx} | {stype} | {created}]\n{preview}", ask_question_pending

    if stype == "CHECKPOINT" and include_system:
        if has_query and not match_query_in_step(obj, raw_query, parsed_query)[0]:
            return None, ask_question_pending
        raw = (obj.get("content") or "").strip()
        cap = 1200 if has_query else 400
        preview = raw[:cap] + ("..." if len(raw) > cap else "")
        return f"\n[Step {idx} | {stype} | {created}]\n{preview}", ask_question_pending

    return None, ask_question_pending


def tool_get_conversation_digest(args: dict[str, Any]) -> str:
    cid = validate_conversation_id(str(get_arg(args, "conversation_id", "conversationId", default="")))
    include_tools = bool(get_arg(args, "include_tools", "includeTools", default=True))
    include_thinking = bool(get_arg(args, "include_thinking", "includeThinking", default=False))
    include_system = bool(get_arg(args, "include_system_messages", "includeSystemMessages", default=True))
    raw_inc_subagents = get_arg(args, "include_subagents", "includeSubagents")
    raw_query = str(get_arg(args, "query", default="") or "").strip()
    has_query = bool(raw_query)
    raw_start = get_arg(args, "start_step", "startStep")
    raw_end = get_arg(args, "end_step", "endStep")
    is_scoped = raw_start is not None or raw_end is not None or has_query
    include_subagents = bool(raw_inc_subagents) if raw_inc_subagents is not None else (not is_scoped)
    has_explicit_start = raw_start is not None or has_query
    start_step = int(raw_start) if raw_start is not None else 0
    end_step = int(raw_end) if raw_end is not None else None
    max_chars = max(2000, min(int(get_arg(args, "max_chars", "maxChars", default=24000)), 100000))

    brain_dir = app_data_dir() / "brain" / cid
    logs_dir = brain_dir / ".system_generated" / "logs"
    transcript_path = logs_dir / "transcript.jsonl"
    full_path = logs_dir / "transcript_full.jsonl"

    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript not found for conversation {cid}: {transcript_path}")

    steps: list[dict[str, Any]] = []
    need_full: set[int] = set()
    planner_result_steps: dict[int, list[int]] = {}
    result_step_subagent: dict[int, str] = {}
    subagent_role_map: dict[str, dict[str, Any]] = {}
    last_planner_idx: int | None = None
    last_planner_tcalls: list[dict[str, Any]] = []
    min_file_idx: int | None = None
    max_file_idx: int | None = None
    total_file_steps = 0
    total_turns = 0
    total_tools = 0
    first_ts = ""
    last_ts = ""

    with transcript_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("step_index", 0)
            t = obj.get("type")
            created = obj.get("created_at") or ""
            if created:
                if not first_ts:
                    first_ts = created
                last_ts = created
            total_file_steps += 1
            if min_file_idx is None or idx < min_file_idx:
                min_file_idx = idx
            if max_file_idx is None or idx > max_file_idx:
                max_file_idx = idx

            if t == "PLANNER_RESPONSE":
                total_turns += 1
                tcalls = obj.get("tool_calls") or []
                total_tools += len(tcalls)
                last_planner_idx = idx
                last_planner_tcalls = tcalls
                planner_result_steps[idx] = []
            elif t == "GENERIC" and last_planner_idx is not None:
                j = len(planner_result_steps[last_planner_idx])
                planner_result_steps[last_planner_idx].append(idx)
                if j < len(last_planner_tcalls):
                    parsed = _parse_subagent_invocation(
                        last_planner_tcalls[j], obj.get("content") or "", last_planner_idx
                    )
                    if parsed:
                        result_step_subagent[idx] = parsed[0]
                        subagent_role_map[parsed[0]] = parsed[1]
            else:
                last_planner_idx = None
                last_planner_tcalls = []

            if idx < start_step:
                continue
            if end_step is not None and idx > end_step:
                continue
            steps.append(obj)
            trunc = obj.get("truncated_fields") or []
            if trunc:
                if t in ("USER_INPUT", "SYSTEM_MESSAGE") and "content" in trunc:
                    need_full.add(idx)
                elif t == "PLANNER_RESPONSE":
                    if (
                        "content" in trunc
                        or ((include_thinking or has_query) and "thinking" in trunc)
                        or ((include_tools or has_query) and "tool_calls" in trunc)
                    ):
                        need_full.add(idx)

    t0 = parse_iso_ts(first_ts)
    t1 = parse_iso_ts(last_ts)
    dur_str = format_duration((t1 - t0).total_seconds() if (t0 and t1) else None)
    dur_part = f", duration={dur_str}" if dur_str else ""
    step_range_str = (
        f"{min_file_idx or 0}..{max_file_idx or 0} "
        f"({total_file_steps} steps, turns={total_turns}, tools={total_tools}{dur_part})"
    )
    app_dir = app_data_dir()
    catalog = _cached_model_catalog(app_dir) if (not is_scoped or include_subagents) else None
    meta_line = f"Conversation: {cid}\nStep range: {step_range_str}"
    subagents: list[str] = []
    subagent_total = 0
    try:
        with open_summaries_db() as conn:
            row = conn.execute(
                "SELECT title, step_count, last_modified_time, workspace_uris, "
                "parent_conversation_id, status, killed, agent_name, nesting_depth "
                "FROM conversation_summaries WHERE conversation_id = ?",
                (cid,),
            ).fetchone()
            if row:
                ws = ", ".join(parse_workspaces(row["workspace_uris"])) or "-"
                parent_cid = row["parent_conversation_id"] or ""
                parent_info = f" | parent={parent_cid}" if parent_cid else ""
                self_ctx = (
                    _cached_inspect_db(
                        cid,
                        app_dir,
                        catalog=catalog,
                        summary_meta={
                            "status": row["status"],
                            "killed": row["killed"],
                            "agent_name": row["agent_name"],
                            "nesting_depth": row["nesting_depth"],
                            "parent_conversation_id": parent_cid,
                        },
                        include_spawned=False,
                    )
                    if not is_scoped
                    else None
                )
                title_str = row["title"] or ""
                if not title_str and parent_cid:
                    parent_map = extract_parent_subagent_map(parent_cid)
                    role_label = format_subagent_role_tag(parent_map.get(cid), self_ctx)
                    title_str = f"[Subagent] {role_label}" if role_label else "(untitled subagent)"
                elif not title_str:
                    title_str = "(untitled)"
                ctx_part = ""
                if not is_scoped:
                    ctx_line = (
                        f"{self_ctx['model']} | {format_compact_context_tag(self_ctx)}"
                        if self_ctx is not None
                        else format_compact_context_tag(None)
                    )
                    ctx_part = f"Context: {ctx_line}\n"
                meta_line = (
                    f"Conversation: {cid}\n"
                    f"Title: {title_str}\n"
                    f"Updated: {compact_ts(row['last_modified_time'])} | Step range: {step_range_str}{parent_info}\n"
                    f"{ctx_part}"
                    f"Workspace: {ws}"
                )
            elif not is_scoped:
                self_ctx = _cached_inspect_db(cid, app_dir, catalog=catalog, include_spawned=False)
                ctx_line = (
                    f"{self_ctx['model']} | {format_compact_context_tag(self_ctx)}"
                    if self_ctx is not None
                    else format_compact_context_tag(None)
                )
                meta_line = f"Conversation: {cid}\nStep range: {step_range_str}\nContext: {ctx_line}"

            sub_rows = conn.execute(
                "SELECT conversation_id, step_count, last_modified_time, status, killed, agent_name, nesting_depth "
                "FROM conversation_summaries "
                "WHERE parent_conversation_id = ? ORDER BY last_modified_time ASC",
                (cid,),
            ).fetchall()
            subagent_total = len(sub_rows)
            if include_subagents:
                for sr in sub_rows:
                    scid = sr["conversation_id"]
                    s_rinfo = subagent_role_map.get(scid.lower())
                    s_ctx = _cached_inspect_db(
                        scid,
                        app_dir,
                        catalog=catalog,
                        summary_meta={
                            "status": sr["status"],
                            "killed": sr["killed"],
                            "agent_name": sr["agent_name"],
                            "nesting_depth": sr["nesting_depth"],
                            "parent_conversation_id": cid,
                            "role": (s_rinfo or {}).get("role", ""),
                            "type_name": (s_rinfo or {}).get("type", ""),
                        },
                        include_spawned=False,
                    )
                    s_metrics = compute_transcript_metrics(scid)
                    rtag = format_subagent_role_tag(s_rinfo, s_ctx)
                    role_part = f" | role={rtag}" if rtag else ""
                    ctx_tag = format_compact_context_tag(s_ctx)
                    if s_metrics:
                        sdur = f", duration={s_metrics['duration']}" if s_metrics.get("duration") else ""
                        subagents.append(
                            f"{scid}{role_part} (steps={s_metrics['steps']}, turns={s_metrics['turns']}, "
                            f"tools={s_metrics['tools']}{sdur} | {ctx_tag})"
                        )
                    else:
                        subagents.append(f"{scid}{role_part} (steps={sr['step_count']} | {ctx_tag})")
    except FileNotFoundError:
        if not is_scoped:
            self_ctx = _cached_inspect_db(cid, app_dir, catalog=catalog, include_spawned=False)
            ctx_line = (
                f"{self_ctx['model']} | {format_compact_context_tag(self_ctx)}"
                if self_ctx is not None
                else format_compact_context_tag(None)
            )
            meta_line = f"{meta_line}\nContext: {ctx_line}"

    header_parts = [meta_line]
    if not is_scoped:
        artifacts = list_artifacts(brain_dir)
        if artifacts:
            header_parts.append("Artifacts:\n" + "\n".join(f"  - {a}" for a in artifacts))
    if subagents:
        header_parts.append("Subagents:\n" + "\n".join(f"  - {s}" for s in subagents))
    elif subagent_total > 0:
        header_parts.append(f"Subagents: {subagent_total} (pass include_subagents=true or use list_conversations to expand)")
    if has_query:
        header_parts.append(f"--- Digest (filtered by query={raw_query!r}) ---")
    else:
        header_parts.append("--- Digest ---")
    header_text = "\n".join(header_parts)

    parsed_query = _parse_step_query(raw_query) if has_query else None
    budget = max(1000, max_chars - len(header_text) - 220)

    def _render_window(window_steps: list[dict[str, Any]], full_map: dict[int, dict[str, Any]]) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        ask_pending = False
        for obj in window_steps:
            idx = obj.get("step_index", 0)
            if idx in full_map and full_map[idx].get("type") == obj.get("type"):
                obj = full_map[idx]
            block, ask_pending = _render_digest_step(
                obj,
                include_tools=include_tools,
                include_thinking=include_thinking,
                include_system=include_system,
                has_query=has_query,
                raw_query=raw_query,
                parsed_query=parsed_query,
                planner_result_steps=planner_result_steps,
                result_step_subagent=result_step_subagent,
                ask_question_pending=ask_pending,
            )
            if block is not None:
                out.append((idx, block))
        return out

    tail_start = 0
    if not has_explicit_start and len(steps) > 40:
        est_used = 0
        est_kept = 0
        target_est = budget + 4000
        for pos in range(len(steps) - 1, -1, -1):
            obj = steps[pos]
            stype = obj.get("type")
            if stype == "GENERIC":
                continue
            block, _ = _render_digest_step(
                obj,
                include_tools=include_tools,
                include_thinking=include_thinking,
                include_system=include_system,
                has_query=False,
                raw_query="",
                parsed_query=None,
                planner_result_steps=planner_result_steps,
                result_step_subagent=result_step_subagent,
                ask_question_pending=False,
            )
            if block is not None:
                est_used += len(block) + 1
                est_kept += 1
                if est_kept >= 2 and est_used > target_est:
                    tail_start = pos
                    break
        while tail_start > 0 and steps[tail_start].get("type") == "GENERIC":
            tail_start -= 1

    window_steps = steps[tail_start:] if tail_start > 0 else steps
    window_need_full = need_full & {s.get("step_index", 0) for s in window_steps} if tail_start > 0 else need_full
    full_map = load_full_steps(full_path, window_need_full)
    rendered_steps = _render_window(window_steps, full_map)

    if tail_start > 0 and sum(len(b) + 1 for _, b in rendered_steps) <= budget:
        full_map = load_full_steps(full_path, need_full)
        rendered_steps = _render_window(steps, full_map)

    if has_query and not rendered_steps:
        return f"{header_text}\nNo steps matched query {raw_query!r} in range {start_step}..{end_step or (max_file_idx or 0)}."

    body_text = "\n".join(block for _, block in rendered_steps)
    if len(header_text) + 1 + len(body_text) <= max_chars or not rendered_steps:
        return f"{header_text}\n{body_text}" if body_text else header_text

    if has_explicit_start:
        kept: list[tuple[int, str]] = []
        used = 0
        for item in rendered_steps:
            if kept and used + len(item[1]) + 1 > budget:
                break
            kept.append(item)
            used += len(item[1]) + 1
        last_kept = kept[-1][0]
        kept_body = "\n".join(block for _, block in kept)
        return (
            f"{header_text}\n{kept_body}\n\n"
            f"... [Digest capped at {max_chars} chars (showing steps {kept[0][0]}..{last_kept}); "
            f"pass start_step={last_kept + 1} to continue]"
        )
    else:
        kept_rev: list[tuple[int, str]] = []
        used = 0
        for item in reversed(rendered_steps):
            if kept_rev and used + len(item[1]) + 1 > budget:
                break
            kept_rev.append(item)
            used += len(item[1]) + 1
        kept = list(reversed(kept_rev))
        first_kept = kept[0][0]
        last_kept = kept[-1][0]
        kept_body = "\n".join(block for _, block in kept)
        banner = (
            f"\n[Showing most recent steps {first_kept}..{last_kept} of {min_file_idx or 0}..{max_file_idx or 0} "
            f"(digest capped at {max_chars} chars); pass start_step=0, end_step={max(0, first_kept - 1)} to inspect earlier steps]"
        )
        return f"{header_text}{banner}\n{kept_body}"


def tool_get_conversation_step(args: dict[str, Any]) -> str:
    cid = validate_conversation_id(str(get_arg(args, "conversation_id", "conversationId", default="")))
    raw_step = get_arg(args, "step_index", "stepIndex")
    raw_steps = get_arg(args, "step_indices", "stepIndices")

    target_list: list[int] = []
    if raw_step is not None:
        target_list.append(int(raw_step))
    if isinstance(raw_steps, list):
        for s in raw_steps:
            idx = int(s)
            if idx not in target_list:
                target_list.append(idx)
    if not target_list:
        raise ValueError("step_index or step_indices is required.")

    use_full = bool(get_arg(args, "full", default=False))
    offset = max(0, int(get_arg(args, "offset", default=0)))
    max_chars = max(500, min(int(get_arg(args, "max_chars", "maxChars", default=16000)), 100000))

    logs_dir = app_data_dir() / "brain" / cid / ".system_generated" / "logs"
    transcript_path = logs_dir / "transcript.jsonl"
    full_path = logs_dir / "transcript_full.jsonl"

    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript not found for conversation {cid}: {transcript_path}")

    target_set = set(target_list)
    chosen_path = full_path if (use_full and full_path.is_file()) else transcript_path
    found_map: dict[int, dict[str, Any]] = {}
    with chosen_path.open(encoding="utf-8") as f:
        for line in f:
            m = STEP_INDEX_PREFIX_RE.match(line)
            if m is not None:
                if int(m.group(1)) not in target_set:
                    continue
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("step_index")
            if idx in target_set:
                found_map[idx] = obj
                if len(found_map) == len(target_set):
                    break

    missing = [i for i in target_list if i not in found_map]
    if len(missing) == len(target_list):
        raise ValueError(f"Step(s) {target_list} not found in {chosen_path.name} for conversation {cid}.")

    need_upgrade = {
        idx
        for idx, obj in found_map.items()
        if ((obj.get("truncated_fields") or obj.get("tool_calls")) and not use_full and full_path.is_file())
    }
    upgraded_map = load_full_steps(full_path, need_upgrade) if need_upgrade else {}

    blocks: list[str] = []
    for idx in target_list:
        if idx not in found_map:
            blocks.append(f"Source: {chosen_path.name} | step_index={idx} | NOT FOUND")
            continue
        if idx in upgraded_map:
            obj = upgraded_map[idx]
            src_note = "transcript_full.jsonl"
        else:
            obj = found_map[idx]
            src_note = chosen_path.name
        formatted = json.dumps(obj, ensure_ascii=False, indent=2)
        blocks.append(f"Source: {src_note} | step_index={idx} | chars={len(formatted)}\n{formatted}")

    combined = "\n\n".join(blocks)
    total_len = len(combined)
    sliced = combined[offset : offset + max_chars]
    more_note = ""
    if offset + max_chars < total_len:
        more_note = f"\n... [truncated at {offset + max_chars}/{total_len} chars; pass offset={offset + max_chars} to read more]"
    return f"{sliced}{more_note}"


def tool_get_conversation_context(args: dict[str, Any]) -> str:
    raw_cid = get_arg(args, "conversation_id", "conversationId")
    raw_cids = get_arg(args, "conversation_ids", "conversationIds")

    target_ids: list[str] = []
    if raw_cid is not None and str(raw_cid).strip():
        target_ids.append(validate_conversation_id(str(raw_cid)))
    if isinstance(raw_cids, list):
        for item in raw_cids:
            if item is None or not str(item).strip():
                continue
            vcid = validate_conversation_id(str(item))
            if vcid not in target_ids:
                target_ids.append(vcid)
    if not target_ids:
        raise ValueError("conversation_id or conversation_ids is required.")
    if len(target_ids) > 50:
        raise ValueError(f"Too many conversation_ids ({len(target_ids)}); maximum is 50 per call.")

    raw_inc_sub = get_arg(args, "include_subagents", "includeSubagents")
    include_subagents = bool(raw_inc_sub) if raw_inc_sub is not None else (len(target_ids) == 1)

    app_dir = app_data_dir()
    catalog = _cached_model_catalog(app_dir)

    summary_map: dict[str, sqlite3.Row] = {}
    sub_rows_by_parent: dict[str, list[sqlite3.Row]] = {}
    try:
        with open_summaries_db() as conn:
            placeholders = ", ".join("?" for _ in target_ids)
            for r in conn.execute(
                "SELECT conversation_id, title, status, killed, agent_name, nesting_depth, parent_conversation_id "
                f"FROM conversation_summaries WHERE conversation_id IN ({placeholders})",
                target_ids,
            ).fetchall():
                summary_map[r["conversation_id"].lower()] = r
            if include_subagents:
                for sr in conn.execute(
                    "SELECT conversation_id, title, status, killed, agent_name, nesting_depth, parent_conversation_id "
                    f"FROM conversation_summaries WHERE parent_conversation_id IN ({placeholders}) "
                    "ORDER BY last_modified_time ASC",
                    target_ids,
                ).fetchall():
                    pcid = (sr["parent_conversation_id"] or "").lower()
                    sub_rows_by_parent.setdefault(pcid, []).append(sr)
    except FileNotFoundError:
        pass

    parent_role_cache: dict[str, dict[str, dict[str, Any]]] = {}
    blocks: list[str] = []
    found_any = False

    for cid in target_ids:
        sr = summary_map.get(cid)
        smeta: dict[str, Any] = {}
        if sr is not None:
            smeta = {
                "status": sr["status"],
                "killed": sr["killed"],
                "agent_name": sr["agent_name"],
                "nesting_depth": sr["nesting_depth"],
                "parent_conversation_id": sr["parent_conversation_id"] or "",
            }
        ctx = _cached_inspect_db(cid, app_dir, catalog=catalog, summary_meta=smeta, include_spawned=include_subagents)
        if ctx is None:
            if len(target_ids) == 1:
                raise FileNotFoundError(
                    f"Conversation database not found for {cid}: {app_dir / 'conversations' / f'{cid}.db'}"
                )
            blocks.append(f"Conversation: {cid}\nError: Conversation database not found ({cid}.db)")
            continue

        found_any = True
        pcid = ctx["parent_conversation_id"].lower() if ctx["parent_conversation_id"] else ""
        if pcid and not ctx["role"]:
            if pcid not in parent_role_cache:
                parent_role_cache[pcid] = extract_parent_subagent_map(pcid)
            p_rinfo = parent_role_cache[pcid].get(cid)
            if p_rinfo:
                ctx["role"] = p_rinfo.get("role") or ctx["role"]
                ctx["type_name"] = p_rinfo.get("type") or ctx["type_name"]

        sub_contexts: list[dict[str, Any]] | None = None
        if include_subagents:
            spawned_map = ctx.get("spawned_subagents") or {}
            ordered_subs: list[str] = []
            sub_smeta_map: dict[str, dict[str, Any]] = {}
            for sub_r in sub_rows_by_parent.get(cid, []):
                scid = sub_r["conversation_id"].lower()
                ordered_subs.append(scid)
                sp_info = spawned_map.get(scid) or {}
                sub_smeta_map[scid] = {
                    "status": sub_r["status"],
                    "killed": sub_r["killed"],
                    "agent_name": sub_r["agent_name"],
                    "nesting_depth": sub_r["nesting_depth"],
                    "parent_conversation_id": cid,
                    "role": sp_info.get("role", ""),
                    "type_name": sp_info.get("type_name", ""),
                }
            for scid, sp_info in spawned_map.items():
                if scid not in sub_smeta_map:
                    ordered_subs.append(scid)
                    sub_smeta_map[scid] = {
                        "parent_conversation_id": cid,
                        "role": sp_info.get("role", ""),
                        "type_name": sp_info.get("type_name", ""),
                    }

            if ordered_subs or raw_inc_sub is True or not pcid:
                sub_contexts = []
                for scid in ordered_subs:
                    s_meta = sub_smeta_map[scid]
                    s_ctx = _cached_inspect_db(
                        scid, app_dir, catalog=catalog, summary_meta=s_meta, include_spawned=False
                    )
                    if s_ctx is None:
                        sub_contexts.append(
                            {"conversation_id": scid, "role": s_meta.get("role", ""), "_missing_db": True}
                        )
                    else:
                        sub_contexts.append(s_ctx)

        blocks.append(render_context_report(ctx, sub_contexts))

    if not found_any:
        raise FileNotFoundError(f"Conversation database(s) not found for: {target_ids}")

    return "\n\n---\n\n".join(blocks)


TOOLS = [
    {
        "name": "list_conversations",
        "description": "List recent Antigravity conversations from conversation_summaries.db, optionally filtered by workspace substring or parent_conversation_id (including subagent roles, turns, tool counts, duration, and context budget usage).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workspace": {
                    "type": "string",
                    "description": "Optional workspace path or project name substring filter (e.g. 'my-project' or '/path/to/workspace').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of conversations to return (default 15, max 100).",
                },
                "include_subagents": {
                    "type": "boolean",
                    "description": "Whether to include subagent conversations in the list (default false).",
                },
                "parent_conversation_id": {
                    "type": "string",
                    "description": "Optional parent conversation UUID to list only its direct subagents with their role and execution metrics.",
                },
            },
        },
    },
    {
        "name": "search_conversations",
        "description": "Full-text search across past conversations (titles, user inputs, agent responses, and subagent transcripts when parent_conversation_id is specified).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrase to search for (use '|' to separate OR alternatives).",
                },
                "workspace": {
                    "type": "string",
                    "description": "Optional workspace path or project name substring filter.",
                },
                "parent_conversation_id": {
                    "type": "string",
                    "description": "Optional parent conversation UUID to search only across its direct subagents (including thinking and tool_calls).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching conversations to return (default 10, max 50).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_conversation_digest",
        "description": "Get a clean, low-token digest of a conversation (metadata, execution metrics, artifacts, subagents with roles/metrics, user requests, tool call summaries with result step indices and MCP arguments, and agent replies), skipping noisy raw tool outputs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "UUID of the conversation to summarize.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional keyword filter (supports space-separated AND tokens or '|'-separated OR alternatives). Searches across content, thinking, and tool_calls (restoring from transcript_full.jsonl) and returns only matching steps.",
                },
                "include_tools": {
                    "type": "boolean",
                    "description": "Include compact one-line summaries of tool calls with their result step_index and inner MCP arguments (default true).",
                },
                "include_thinking": {
                    "type": "boolean",
                    "description": "Include full model internal thinking blocks restored from transcript_full.jsonl (default false; auto-enabled on steps where thinking matches query).",
                },
                "include_system_messages": {
                    "type": "boolean",
                    "description": "Include subagent notifications and compaction checkpoints, excluding noisy timer/task wakeups (default true).",
                },
                "include_subagents": {
                    "type": "boolean",
                    "description": "Whether to expand the full subagent list and metrics in the header (defaults to true for full overview, false when start_step/end_step/query is set).",
                },
                "start_step": {
                    "type": "integer",
                    "description": "Optional starting step_index (inclusive).",
                },
                "end_step": {
                    "type": "integer",
                    "description": "Optional ending step_index (inclusive).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 24000, max 100000). When start_step and query are omitted and the digest exceeds max_chars, the most recent steps within budget are returned.",
                },
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "get_conversation_step",
        "description": "Read one or more steps (including raw tool outputs or full untruncated content/thinking/tool_calls) by step_index or step_indices from transcript.jsonl or transcript_full.jsonl.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "UUID of the conversation.",
                },
                "step_index": {
                    "type": "integer",
                    "description": "Single step_index to inspect (provide step_index or step_indices).",
                },
                "step_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Batch list of step_index values to inspect in a single call.",
                },
                "full": {
                    "type": "boolean",
                    "description": "Read directly from transcript_full.jsonl (default false; auto-upgrades if truncated_fields is present).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset for paginating very large steps (default 0).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 16000, max 100000).",
                },
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "get_conversation_context",
        "description": "Inspect exact context window usage, 256k compaction budget watermark, per-category token breakdown (user, agent, tool calls, system, rules, skills, MCP tools), and checkpoint compaction status from conversations/<id>.db, optionally including all spawned subagents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "Single conversation UUID to inspect (provide conversation_id or conversation_ids).",
                },
                "conversation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Batch list of conversation UUIDs (e.g. multiple subagents) to inspect in a single call.",
                },
                "include_subagents": {
                    "type": "boolean",
                    "description": "Include context usage and compaction budget summaries for direct subagents (defaults to true when inspecting a single conversation, false for multi-ID batches).",
                },
            },
        },
    },
]

TOOL_HANDLERS = {
    "list_conversations": tool_list_conversations,
    "search_conversations": tool_search_conversations,
    "get_conversation_digest": tool_get_conversation_digest,
    "get_conversation_step": tool_get_conversation_step,
    "get_conversation_context": tool_get_conversation_context,
}


def handle_request(msg: dict[str, Any]) -> dict[str, Any] | None:
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if req_id is None:
        return None

    if method == "initialize":
        client_version = params.get("protocolVersion") or "2024-11-05"
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "session-context", "version": "1.2.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        try:
            text = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": text}], "isError": False},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {type(exc).__name__}: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_request(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
