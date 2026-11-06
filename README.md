# session-context

A lightweight **Antigravity Plugin** that exposes a zero-dependency Python Stdio MCP server and routing rules for cross-conversation context lookup and distillation.

## Features

- **`list_conversations`**: Queries `~/.gemini/antigravity/conversation_summaries.db` (read-only) to list recent conversations by workspace or parent conversation (for subagents).
- **`search_conversations`**: Uses SQLite FTS5 (`conversations_fts`) to search across conversation titles, workspaces, user inputs, and agent replies with highlighted match snippets.
- **`get_conversation_digest`**: Parses `transcript.jsonl` (automatically upgrading truncated user/agent messages from `transcript_full.jsonl`) and returns a compact digest of user prompts, tool call one-liners, `ask_question` answers, agent responses, artifacts, and subagents—skipping bulky `GENERIC` tool outputs to save 85%+ context tokens.
- **`get_conversation_step`**: Fetches a single step by `step_index` with pagination when raw tool output or full JSON inspection is needed.

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
