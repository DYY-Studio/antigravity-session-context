# Cross-Conversation Context Lookup (`session-context`)

When the user asks to reference, search, or inspect another conversation (by `conversation-id`, topic/keyword, or workspace/recent history):

1. **Prefer `session-context_mcp` MCP tools over raw file reads**:
   - `list_conversations({ workspace?, limit?, include_subagents?, parent_conversation_id? })`: List recent conversations.
   - `search_conversations({ query, workspace?, limit? })`: Run full-text FTS5 searches across titles, user requests, and agent responses (returns matching `conversation_id`, snippets, and `matched_steps`).
   - `get_conversation_digest({ conversation_id, start_step?, end_step?, max_chars?, include_tools?, include_thinking?, include_system_messages? })`: Retrieve a distilled, low-token summary of turns, tool calls (annotated with their `[Step N]` result index), artifacts, and subagents.
   - `get_conversation_step({ conversation_id, step_index, full?, offset?, max_chars? })`: Inspect the raw tool output or full untruncated payload of a specific `step_index`.
2. **Avoid reading raw `transcript.jsonl` or `transcript_full.jsonl` directly with `read_file`** unless the `session-context` MCP tools are unavailable, as raw transcripts include large `GENERIC` tool outputs that rapidly exhaust the context window.
