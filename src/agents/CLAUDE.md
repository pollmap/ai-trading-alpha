# Agents Module (Agent Execution Protocol) Rules

## Role

Define single/multi agents and the benchmark orchestrator for parallel execution.

## Rules

- `orchestrator.py` is the ONLY external entry point — other modules do not call agents directly
- All agent combinations (4 models x 2 architectures) must run on the SAME MarketSnapshot
- Each combination has fully independent PortfolioState — absolutely no sharing
- Buy-and-hold baseline is always included as the 9th "agent"
- GPT-4o-mini is included as reference baseline (not a core analysis target)

## Multi-Agent Pipeline (LangGraph)

- `graph.py` defines the StateGraph
- State is TypedDict (NOT dataclass — LangGraph compatibility)
- 4 analyst nodes MUST run in parallel (fan-out -> fan-in pattern)
- Risk Manager VETO -> conditional edge back to Trader node
- Each node uses the Runner's LLM adapter from state

## Error Isolation

- One agent combination failure -> remaining agents continue normally
- Failed combination: HOLD for that cycle + error logging + normal retry next cycle
- Timeout: single 30s, multi 120s total
