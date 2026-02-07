# LLM Module Rules

## Role

Provide unified adapters for 4 LLM providers (DeepSeek, Gemini, Claude, GPT-4o-mini).

## Rules

- All adapters implement BaseLLMAdapter
- **Every LLM call must run inside cost_tracker.track() context manager**
- **Every LLM call must be recorded via call_logger (prompt + raw response in DB)**
- Prompt structure: system_prompt (static, cacheable) + user_prompt (dynamic)
- Parse failure x3 -> return HOLD signal, log full raw response
- Timeout: single agent 30s, multi-agent node 60s
- temperature: 0.0 fixed for benchmark fairness (overridable via config)

## Provider-Specific Notes

### DeepSeek
- SDK: `openai.AsyncOpenAI(base_url="https://api.deepseek.com")`
- Model: `"deepseek-reasoner"` (R1) or `"deepseek-chat"` (V3)
- Extract CoT from `reasoning_content` field -> store in audit trail
- JSON mode: `response_format={"type": "json_object"}`

### Gemini
- SDK: `google.genai.Client()`
- Model: `"gemini-2.5-pro-preview-06-05"`
- JSON mode: `response_mime_type="application/json"` + `response_schema`
- 1M token context: can pass all analyst reports at once in multi-agent

### Claude
- SDK: `anthropic.AsyncAnthropic()`
- Model: `"claude-sonnet-4-5-20250514"`
- Extended thinking: activate for complex analysis
- Prompt caching: `cache_control={"type": "ephemeral"}` on system prompt (90% cost reduction)
- Structured output: XML tag-based (`<action>`, `<weight>`, `<confidence>`, `<reasoning>`)

### GPT-4o-mini (Reference Baseline)
- SDK: `openai.AsyncOpenAI()` — no base_url change needed
- Model: `"gpt-4o-mini"`
- JSON mode: `response_format={"type": "json_object"}`
- Structurally identical to DeepSeek adapter

## Prompt Template Rules

- Files under prompt_templates/ organized by role
- All prompts defined as functions (not string constants): `def build_prompt(snapshot, portfolio) -> str`
- Few-shot examples: max 2 per prompt (token conservation)
- Output format explicitly specified (JSON schema or XML tag definitions)
