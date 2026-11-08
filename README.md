# session-context

A lightweight **Antigravity Plugin** that exposes a zero-dependency Python Stdio MCP server and routing rules for cross-conversation and subagent context lookup, search, and distillation.

## Features

- **`list_conversations`**: Queries `~/.gemini/antigravity/conversation_summaries.db` (read-only) to list recent conversations by workspace or `parent_conversation_id` (automatically extracting subagent roles and computing `steps`, `turns`, `tools`, and `duration`).
- **`search_conversations`**: Uses SQLite FTS5 (`conversations_fts`) and transcript scanning (including `thinking` and `tool_calls` for subagents when `parent_conversation_id` is specified) with `|` OR-query support and `matched_steps` indices.
- **`get_conversation_digest`**: Parses `transcript.jsonl` (automatically restoring truncated `content`, `thinking`, and `tool_calls` from `transcript_full.jsonl`) and returns a compact digest of execution metrics, artifacts, subagents with roles/metrics, user prompts, tool call summaries (including inner MCP `Arguments` and spawned subagent IDs), `ask_question` answers, and agent responses. Supports in-conversation `query` filtering across `content`, `thinking`, and `tool_calls`.
- **`get_conversation_step`**: Fetches one (`step_index`) or multiple (`step_indices`) steps in a single call with automatic `transcript_full.jsonl` upgrade and pagination.

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
