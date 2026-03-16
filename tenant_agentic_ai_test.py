import json
from ollama import ChatResponse, chat
# Example code from, modified to work with Ollama
# https://medium.com/@abel.ncm/build-your-first-agentic-ai-from-scratch-in-minutes-38f26a6d7f7f

MODEL = "qwen3.5:4b"
BAN_PHRASES = ["import", "os", "sys", "__", "eval", "exec"]
 
TENANTS = [
    {"name": "Alice Johnson", "apartment": "12A", "owner": True},
    {"name": "Bob Smith",     "apartment": "8C",  "owner": False},
    {"name": "Carla Mendes",  "apartment": "15B", "owner": True},
]
 
# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------
 
def tenant_lookup(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"found": True, "tenant": t}
    return {"found": False}
 
def tenant_list():
    return {"tenants": TENANTS}
 
def tenant_is_owner(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"name": name, "is_owner": t["owner"]}
    return {"error": "tenant not found"}
 
def calculator(expr: str):
    if any(b in expr.lower() for b in BAN_PHRASES):
        return {"error": "blocked unsafe expression"}
    try:
        return {"result": eval(expr)}
    except Exception as e:
        return {"error": str(e)}
 
# Dispatch table — normalises argument key differences between schema and function
TOOLS = {
    "calculator":      lambda args: calculator(args.get("expr", "")),
    "tenant_lookup":   lambda args: tenant_lookup(args.get("name", "")),
    "tenant_list":     lambda args: tenant_list(),
    "tenant_is_owner": lambda args: tenant_is_owner(args.get("name", "")),
}

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate a math expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "Math expression, e.g. '500 * 1.08'"}
                },
                "required": ["expr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_lookup",
            "description": "Look up a condo tenant by their full name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_list",
            "description": "Return a list of all tenants in the building.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_is_owner",
            "description": "Check whether a named tenant is an owner (vs renter).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        }
    },
]
 
# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------
 
def llm(messages: list) -> ChatResponse:
    return chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS)

def llm_plain(messages: list) -> ChatResponse:
    """For planner and reflector - no tools, just text."""
    return chat(model=MODEL, messages=messages)
 
def agent_loop(goal: str, max_steps: int = 6) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful agent with access to tools. "
                "Always use the provided tools to look up real data rather than guessing. "
                "When you have a definitive answer, state it clearly in plain text."
            )
        },
        {"role": "user", "content": goal}
    ]
 
    for step in range(max_steps):
        print(f"  --- step {step + 1} ---")
 
        response = llm(messages)
        msg = response.message
 
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls or [],
        })
 
        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or {}
                print(f"  [tool  ] {name}({args})")
 
                result = TOOLS[name](args) if name in TOOLS else {"error": f"unknown tool: {name}"}
                print(f"  [result] {result}")
 
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })
            # Tool results appended — loop continues to next LLM call
        else:
            # No tool calls → model has finished reasoning
            print(f"  [answer] {msg.content}")
            return msg.content or ""
 
    return "Max steps reached without a final answer."
 
# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
 
def plan(goal: str) -> list[str]:
    prompt = (
        "Break the following task into 2–4 clear, actionable steps. "
        "Return ONLY a JSON array of strings with no explanation or markdown.\n\n"
        f"Task: {goal}"
    )
    response = llm_plain([{"role": "user", "content": prompt}])
    text = response.message.content.strip()
 
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
 
    try:
        steps = json.loads(text)
        if isinstance(steps, list):
            return [str(s) for s in steps]
    except json.JSONDecodeError:
        pass
 
    return [line.strip() for line in text.splitlines() if line.strip()]
 
# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------
 
def reflect(step: str, answer: str) -> bool:
    prompt = (
        f"Step: {step}\n"
        f"Answer: {answer}\n\n"
        "Is this answer correct and complete for the step above? "
        "Reply with ONLY the word YES or NO."
    )
    response = llm_plain([{"role": "user", "content": prompt}])
    verdict = response.message.content.strip().upper()
    passed = verdict.startswith("YES")
    print(f"  [reflect] {verdict} → {'pass' if passed else 'RETRY'}")
    return passed
 
# ---------------------------------------------------------------------------
# Orchestrator - passes prior results explicitly instead of via global memory
# ---------------------------------------------------------------------------
 
def multi_agent(goal: str, max_retries: int = 2) -> list[str]:
    print("\n=== PLANNING ===")
    steps = plan(goal)
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
 
    results = []
    original_steps = list(steps)
 
    for step in steps:
        print(f"\n=== WORKER: {step} ===")
        answer = None
 
        # Build context from prior steps so the model isn't flying blind
        context = ""
        if results:
            context = "Previous step results:\n" + "\n".join(
                f"  - {s}: {r}" for s, r in zip(original_steps, results)
            ) + "\n\n"
 
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"\n  [retry {attempt}/{max_retries}]")
                step = f"{step}  (previous attempt gave wrong answer: {answer!r} — try again using tools)"
 
            answer = agent_loop(context + step)
 
            print(f"\n=== REVIEWER ===")
            if reflect(step, answer):
                break
 
        results.append(answer)
 
    print("\n=== FINAL RESULTS ===")
    for step, result in zip(original_steps, results):
        print(f"  • {step}\n    → {result}\n")
 
    return results

if __name__ == "__main__":
    # Try condo questions:
    # goal = "Is Bob Smith an owner?"
    # goal = "List every tenant in the building."
    # goal = "Which apartment does Alice Johnson live in?"

    goal = "Check if Carla Mendes is an owner and calculate 500*1.08"
    print(multi_agent(goal))