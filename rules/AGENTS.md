# Cross-Conversation Context Lookup (`session-context`)

When the user asks to reference, search, or inspect another conversation (by `conversation-id`, topic/keyword, or workspace/recent history):

1. **Prefer `session-context` MCP tools over raw file reads**:
   - Use `list_conversations` to find recent conversations (optionally filtered by `workspace` or `parent_conversation_id`).
   - Use `search_conversations` to run full-text FTS5 searches across titles, user requests, and agent responses when no `conversation-id` is given.
   - Use `get_conversation_digest` to retrieve a distilled, low-token summary of a conversation's turns, tool call summaries, artifacts, and subagents.
   - Use `get_conversation_step` only when you need to inspect the raw output or full untruncated payload of a specific `step_index`.
2. **Avoid reading raw `transcript.jsonl` or `transcript_full.jsonl` directly with `read_file`** unless the `session-context` MCP tools are unavailable, as raw transcripts include large `GENERIC` tool outputs that rapidly exhaust the context window.
