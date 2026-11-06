#!/usr/bin/env python3
"""Lightweight MCP server for Antigravity cross-conversation context lookup and digest."""

import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
from urllib.parse import unquote, urlparse

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
USER_REQ_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.DOTALL)


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


def extract_user_text(raw: str) -> str:
    m = USER_REQ_RE.search(raw or "")
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", raw or "", flags=re.DOTALL)
    cleaned = re.sub(r"<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def summarize_tool_call(tc: dict[str, Any]) -> str:
    name = tc.get("name", "unknown")
    args = tc.get("args") or {}
    if not isinstance(args, dict):
        return f"{name}()"
    highlights = []
    for k in ("Path", "CommandLine", "toolSummary", "query", "queries", "Role", "TypeName", "Recipient", "ServerName", "ToolName", "Action"):
        if k in args and args[k] is not None:
            val = str(args[k]).strip().strip('"')
            if len(val) > 80:
                val = val[:77] + "..."
            highlights.append(f"{k}={val}")
    if not highlights and args:
        first_k = next(iter(args))
        val = str(args[first_k]).strip().strip('"')
        if len(val) > 80:
            val = val[:77] + "..."
        highlights.append(f"{first_k}={val}")
    return f"{name}({', '.join(highlights)})"


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
            clauses.append("workspace_uris LIKE ?")
            params.append(f"%{workspace.replace(chr(92), '/')}%")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT conversation_id, title, step_count, last_modified_time, "
            "workspace_uris, parent_conversation_id, status "
            f"FROM conversation_summaries {where_sql} "
            "ORDER BY last_modified_time DESC LIMIT ?"
        )
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return "No matching conversations found."

    lines = [f"Found {len(rows)} conversation(s):"]
    for r in rows:
        cid = r["conversation_id"]
        title = r["title"] or "(untitled)"
        ws = ", ".join(parse_workspaces(r["workspace_uris"])) or "-"
        sub_n = sub_counts.get(cid, 0)
        sub_tag = f" | subagents={sub_n}" if sub_n else ""
        parent_tag = f" | parent={r['parent_conversation_id']}" if r["parent_conversation_id"] else ""
        lines.append(
            f"- {cid} | {r['last_modified_time']} | steps={r['step_count']}{sub_tag}{parent_tag}\n"
            f"  title: {title}\n"
            f"  workspace: {ws}"
        )
    return "\n".join(lines)


def build_fts_query(raw_query: str) -> str:
    tokens = [t for t in re.split(r"\s+", raw_query.strip()) if t]
    if not tokens:
        raise ValueError("Search query must not be empty.")
    escaped = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " ".join(escaped)


def find_matching_steps(cid: str, raw_query: str, max_shown: int = 12) -> str:
    tokens = [t.lower() for t in re.split(r"\s+", raw_query.strip()) if t]
    if not tokens:
        return ""
    transcript_path = app_data_dir() / "brain" / cid / ".system_generated" / "logs" / "transcript.jsonl"
    if not transcript_path.is_file():
        return ""
    all_matches: list[int] = []
    any_matches: list[int] = []
    try:
        with transcript_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                stype = obj.get("type")
                if stype not in ("USER_INPUT", "PLANNER_RESPONSE", "SYSTEM_MESSAGE"):
                    continue
                idx = obj.get("step_index")
                if idx is None:
                    continue
                hay = ((obj.get("content") or "") + " " + json.dumps(obj.get("tool_calls") or [], ensure_ascii=False)).lower()
                if all(t in hay for t in tokens):
                    all_matches.append(idx)
                elif any(t in hay for t in tokens):
                    any_matches.append(idx)
    except Exception:
        return ""
    chosen = all_matches if all_matches else any_matches
    if not chosen:
        return ""
    shown = ", ".join(str(i) for i in chosen[:max_shown])
    if len(chosen) > max_shown:
        return f"[{shown}, ... ({len(chosen)} steps)]"
    return f"[{shown}]"


def tool_search_conversations(args: dict[str, Any]) -> str:
    raw_query = str(get_arg(args, "query", default="") or "").strip()
    workspace = str(get_arg(args, "workspace", default="") or "").strip()
    limit = max(1, min(int(get_arg(args, "limit", default=10)), 50))
    fts_query = build_fts_query(raw_query)

    with open_summaries_db() as conn:
        clauses = ["conversations_fts MATCH ?"]
        params: list[Any] = [fts_query]
        if workspace:
            clauses.append("s.workspace_uris LIKE ?")
            params.append(f"%{workspace.replace(chr(92), '/')}%")

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

    if not rows:
        return f"No conversations matched query: {raw_query!r}"

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

    lines = [f"Matched {len(grouped)} conversation(s) for {raw_query!r}:"]
    for cid, info in grouped.items():
        r = info["meta"]
        title = r["title"] or "(untitled)"
        ws = ", ".join(parse_workspaces(r["workspace_uris"] or "")) or "-"
        parent_tag = f" | parent={r['parent_conversation_id']}" if r["parent_conversation_id"] else ""
        step_hits = find_matching_steps(cid, raw_query)
        step_hits_line = f"\n  matched_steps: {step_hits}" if step_hits else ""
        lines.append(
            f"- {cid} | {r['last_modified_time']} | steps={r['step_count']}{parent_tag}\n"
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


def tool_get_conversation_digest(args: dict[str, Any]) -> str:
    cid = validate_conversation_id(str(get_arg(args, "conversation_id", "conversationId", default="")))
    include_tools = bool(get_arg(args, "include_tools", "includeTools", default=True))
    include_thinking = bool(get_arg(args, "include_thinking", "includeThinking", default=False))
    include_system = bool(get_arg(args, "include_system_messages", "includeSystemMessages", default=True))
    raw_start = get_arg(args, "start_step", "startStep")
    raw_end = get_arg(args, "end_step", "endStep")
    has_explicit_start = raw_start is not None
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
    last_planner_idx: int | None = None
    min_file_idx: int | None = None
    max_file_idx: int | None = None
    total_file_steps = 0

    with transcript_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("step_index", 0)
            t = obj.get("type")
            total_file_steps += 1
            if min_file_idx is None or idx < min_file_idx:
                min_file_idx = idx
            if max_file_idx is None or idx > max_file_idx:
                max_file_idx = idx

            if t == "PLANNER_RESPONSE":
                last_planner_idx = idx
                planner_result_steps[idx] = []
            elif t == "GENERIC" and last_planner_idx is not None:
                planner_result_steps[last_planner_idx].append(idx)
            else:
                last_planner_idx = None

            if idx < start_step:
                continue
            if end_step is not None and idx > end_step:
                continue
            steps.append(obj)
            trunc = obj.get("truncated_fields") or []
            if t in ("USER_INPUT", "PLANNER_RESPONSE") and "content" in trunc:
                need_full.add(idx)

    step_range_str = f"{min_file_idx or 0}..{max_file_idx or 0} ({total_file_steps} steps)"
    meta_line = f"Conversation: {cid}\nStep range: {step_range_str}"
    subagents: list[str] = []
    try:
        with open_summaries_db() as conn:
            row = conn.execute(
                "SELECT title, step_count, last_modified_time, workspace_uris, parent_conversation_id "
                "FROM conversation_summaries WHERE conversation_id = ?",
                (cid,),
            ).fetchone()
            if row:
                ws = ", ".join(parse_workspaces(row["workspace_uris"])) or "-"
                parent_info = f" | parent={row['parent_conversation_id']}" if row["parent_conversation_id"] else ""
                meta_line = (
                    f"Conversation: {cid}\n"
                    f"Title: {row['title'] or '(untitled)'}\n"
                    f"Updated: {row['last_modified_time']} | Step range: {step_range_str}{parent_info}\n"
                    f"Workspace: {ws}"
                )
            sub_rows = conn.execute(
                "SELECT conversation_id, step_count, last_modified_time FROM conversation_summaries "
                "WHERE parent_conversation_id = ? ORDER BY last_modified_time ASC",
                (cid,),
            ).fetchall()
            subagents = [f"{sr['conversation_id']} (steps={sr['step_count']})" for sr in sub_rows]
    except FileNotFoundError:
        pass

    full_map = load_full_steps(full_path, need_full)

    header_parts = [meta_line]
    artifacts = list_artifacts(brain_dir)
    if artifacts:
        header_parts.append("Artifacts:\n" + "\n".join(f"  - {a}" for a in artifacts))
    if subagents:
        header_parts.append("Subagents:\n" + "\n".join(f"  - {s}" for s in subagents))
    header_parts.append("--- Digest ---")
    header_text = "\n".join(header_parts)

    rendered_steps: list[tuple[int, str]] = []
    ask_question_pending = False
    for obj in steps:
        idx = obj.get("step_index", 0)
        stype = obj.get("type")
        created = obj.get("created_at", "")
        full_note = ""
        if idx in full_map:
            obj = full_map[idx]
            full_note = " [restored from transcript_full.jsonl]"

        if stype == "USER_INPUT":
            ask_question_pending = False
            text = extract_user_text(obj.get("content", ""))
            rendered_steps.append((idx, f"\n[Step {idx} | USER | {created}]{full_note}\n{text}"))

        elif stype == "PLANNER_RESPONSE":
            ask_question_pending = False
            parts = []
            if include_thinking and obj.get("thinking"):
                parts.append(f"<thinking>\n{obj['thinking'].strip()}\n</thinking>")
            tcalls = obj.get("tool_calls") or []
            if include_tools and tcalls:
                res_indices = planner_result_steps.get(idx, [])
                summaries = []
                for j, tc in enumerate(tcalls):
                    sc = summarize_tool_call(tc)
                    if j < len(res_indices):
                        summaries.append(f"[Step {res_indices[j]}] {sc}")
                    else:
                        summaries.append(sc)
                parts.append("Tools: " + "; ".join(summaries))
            for tc in tcalls:
                if tc.get("name") == "ask_question":
                    ask_question_pending = True
            content = (obj.get("content") or "").strip()
            if content:
                parts.append(content)
            if parts:
                rendered_steps.append((idx, f"\n[Step {idx} | AGENT | {created}]{full_note}\n" + "\n".join(parts)))

        elif stype == "GENERIC" and ask_question_pending:
            ask_question_pending = False
            ans = (obj.get("content") or "").strip()
            if ans:
                rendered_steps.append((idx, f"\n[Step {idx} | USER_ANSWER | {created}]\n{ans}"))

        elif stype in ("SYSTEM_MESSAGE", "CHECKPOINT") and include_system:
            raw = (obj.get("content") or "").strip()
            preview = raw[:400] + ("..." if len(raw) > 400 else "")
            rendered_steps.append((idx, f"\n[Step {idx} | {stype} | {created}]\n{preview}"))

    body_text = "\n".join(block for _, block in rendered_steps)
    if len(header_text) + 1 + len(body_text) <= max_chars or not rendered_steps:
        return f"{header_text}\n{body_text}" if body_text else header_text

    budget = max(1000, max_chars - len(header_text) - 220)
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
    if raw_step is None:
        raise ValueError("step_index is required.")
    target_idx = int(raw_step)
    use_full = bool(get_arg(args, "full", default=False))
    offset = max(0, int(get_arg(args, "offset", default=0)))
    max_chars = max(500, min(int(get_arg(args, "max_chars", "maxChars", default=16000)), 100000))

    logs_dir = app_data_dir() / "brain" / cid / ".system_generated" / "logs"
    transcript_path = logs_dir / "transcript.jsonl"
    full_path = logs_dir / "transcript_full.jsonl"

    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript not found for conversation {cid}: {transcript_path}")

    chosen_path = full_path if (use_full and full_path.is_file()) else transcript_path
    found = None
    with chosen_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("step_index") == target_idx:
                found = obj
                break

    if found is None:
        raise ValueError(f"Step {target_idx} not found in {chosen_path.name} for conversation {cid}.")

    source_note = chosen_path.name
    trunc = found.get("truncated_fields") or []
    if trunc and not use_full and full_path.is_file():
        full_step = load_full_steps(full_path, {target_idx}).get(target_idx)
        if full_step is not None:
            found = full_step
            source_note = "transcript_full.jsonl (auto-upgraded due to truncated_fields)"

    formatted = json.dumps(found, ensure_ascii=False, indent=2)
    total_len = len(formatted)
    sliced = formatted[offset : offset + max_chars]
    more_note = ""
    if offset + max_chars < total_len:
        more_note = f"\n... [truncated at {offset + max_chars}/{total_len} chars; pass offset={offset + max_chars} to read more]"
    return f"Source: {source_note} | step_index={target_idx} | total_chars={total_len}\n{sliced}{more_note}"


TOOLS = [
    {
        "name": "list_conversations",
        "description": "List recent Antigravity conversations from conversation_summaries.db, optionally filtered by workspace substring or parent_conversation_id.",
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
                    "description": "Optional parent conversation UUID to list only its direct subagents.",
                },
            },
        },
    },
    {
        "name": "search_conversations",
        "description": "Full-text search across past conversations (titles, user inputs, agent responses) using the local SQLite FTS5 index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrase to search for.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Optional workspace path or project name substring filter.",
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
        "description": "Get a clean, low-token digest of a conversation (metadata, artifacts, subagents, user requests, tool call summaries with result step indices, and agent replies), skipping noisy raw tool outputs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "UUID of the conversation to summarize.",
                },
                "include_tools": {
                    "type": "boolean",
                    "description": "Include compact one-line summaries of tool calls with their result step_index (default true).",
                },
                "include_thinking": {
                    "type": "boolean",
                    "description": "Include model internal thinking blocks (default false).",
                },
                "include_system_messages": {
                    "type": "boolean",
                    "description": "Include system/subagent notifications and compaction checkpoints (default true).",
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
                    "description": "Maximum characters to return (default 24000, max 100000). When start_step is omitted and the digest exceeds max_chars, the most recent steps within budget are returned.",
                },
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "get_conversation_step",
        "description": "Read a single step (including raw tool outputs or full untruncated content) by step_index from transcript.jsonl or transcript_full.jsonl.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "UUID of the conversation.",
                },
                "step_index": {
                    "type": "integer",
                    "description": "Exact step_index to inspect.",
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
            "required": ["conversation_id", "step_index"],
        },
    },
]

TOOL_HANDLERS = {
    "list_conversations": tool_list_conversations,
    "search_conversations": tool_search_conversations,
    "get_conversation_digest": tool_get_conversation_digest,
    "get_conversation_step": tool_get_conversation_step,
}


def handle_request(msg: dict[str, Any]) -> dict[str, Any] | None:
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if req_id is None:
        return None

    if method == "initialize":
        client_Version = params.get("protocolVersion") or "2024-11-05"
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_Version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "session-context", "version": "1.0.0"},
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
