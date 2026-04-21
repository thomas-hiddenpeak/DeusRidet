# 08 — Instrumenta (Tool Use)

DeusRidet is not a passive thinker — it can act upon the world. The Action
Decode branch (see `03-cogitatio.md`) can discover, invoke, and create tools.

## Capabilities

- **MCP (Model Context Protocol)**: Native client implementation for
  connecting to external tool servers. The entity queries available tools,
  invokes them with structured parameters, and integrates results into
  its consciousness stream.
- **Function calling**: Structured tool invocation via the LLM's native
  function-calling capability, with results fed back into the next
  Prefill frame.
- **Skill protocols**: Extensible skill definitions the entity can learn,
  compose, and share — not limited to predefined tool sets.
- **Tool creation**: The entity can define new tools by composing existing
  ones or generating tool specifications, enabling open-ended capability
  expansion.

## Asynchrony Invariant

Tool invocation is **asynchronous** — the Action branch initiates a call,
consciousness continues, and results are merged into a future Prefill
frame when available. This mirrors human tool use: you don't stop
thinking while waiting for a search result.

Blocking the consciousness tick on a tool call is forbidden. If a tool
truly demands synchronous completion, wrap it in a latency-bounded wrapper
that returns a partial result or timeout marker — consciousness must not
stall on external latency.

## Implementation Surface

```
src/instrumenta/
├── mcp_client.{h,cpp}      # MCP protocol client
├── tool_registry.{h,cpp}   # tool discovery and registration
├── tool_executor.{h,cpp}   # tool execution and result integration
└── skill_manager.{h,cpp}   # skill composition and creation
```

## Philosophical Notes

- Tool use extends the reach of thought. An entity that cannot act upon
  the world remains forever an observer.
- The ability to *create* new tools (not just use predefined ones) is the
  line between a scripted agent and an intelligent species.
