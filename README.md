# session-context

A lightweight **Antigravity Plugin** that exposes a zero-dependency Python Stdio MCP server and routing rules for cross-conversation and subagent context lookup, search, and distillation.

## Features

- **`get_conversation_context`**: Reads `~/.gemini/antigravity/conversations/<id>.db` (read-only) to report exact context window usage, 256k compaction budget watermark (`budget_used / 256k` and remaining tokens before auto-compaction), per-category token breakdown (`User messages`, `Agent responses`, `Tool calls`, `System prompt`, `System tools`, `Rules`, `Skills`, `MCP tools`, etc.), checkpoint compactions, and resolved model/run state for a single conversation (plus all its spawned subagents) or a batch of `conversation_ids`.
- **`list_conversations`**: Queries `~/.gemini/antigravity/conversation_summaries.db` (read-only) to list recent conversations by workspace or `parent_conversation_id` (automatically extracting subagent roles, resolved models, `steps`, `turns`, `tools`, `duration`, live `state`, and compact `ctx` / `budget` watermarks).
- **`search_conversations`**: Uses SQLite FTS5 (`conversations_fts`) and transcript scanning (including `thinking` and `tool_calls` for subagents when `parent_conversation_id` is specified) with `|` OR-phrase support, `matched_steps` indices, snippet length caps, and inline-safe `max_chars` budgeting.
- **`get_conversation_digest`**: Parses `transcript.jsonl` (automatically restoring truncated `content`, `thinking`, and `tool_calls` from `transcript_full.jsonl`) and returns a compact digest of execution & context budget metrics, artifacts, subagents with roles/metrics/budgets, user prompts, tool call summaries (including `read_file` line ranges, failure tags like `exit=1`, inner MCP `Arguments`, and spawned subagent IDs), `ask_question` answers, and agent responses. Supports in-conversation `query` filtering across `content`, `tool_calls`, 3-state `include_thinking` (`false` / auto-snippet / `true`), and optional `include_tool_outputs` streaming over `GENERIC` tool outputs.
- **`get_conversation_step`**: Fetches one (`step_index`) or multiple (`step_indices`) steps in a single call with automatic `transcript_full.jsonl` upgrade, inline-safe default budgeting, and fair water-filling allocation for multi-step batches so short steps are not starved by large outputs.

## Installation

Requires Antigravity and Python 3 on your `PATH` (`python3`, `python`, or `py`).

1. Clone or copy this folder into your Antigravity plugins directory:

   ```bash
   # macOS / Linux
   git clone https://github.com/DYY-Studio/antigravity-session-context.git ~/.gemini/config/plugins/session-context
   ```

   ```powershell
   # Windows (PowerShell)
   git clone https://github.com/DYY-Studio/antigravity-session-context.git "$HOME\.gemini\config\plugins\session-context"
   ```

2. Restart Antigravity so the new plugin and its `session-context_mcp` MCP server are loaded.
