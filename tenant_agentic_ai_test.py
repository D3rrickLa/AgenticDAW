import json

from ollama import ChatResponse, chat

# Example code from, modified to work with Ollama
# https://medium.com/@abel.ncm/build-your-first-agentic-ai-from-scratch-in-minutes-38f26a6d7f7f

MODEL = "qwen2.5-coder:3b"
BAN_PRHASES  = ["import", "os", "sys", "__", "eval", "exec"]

TENANTS = [
    {"name": "Alice Johnson", "apartment": "12A", "owner": True},
    {"name": "Bob Smith", "apartment": "8C", "owner": False},
    {"name": "Carla Mendes", "apartment": "15B", "owner": True},
]

# TOOL Functions
def tenant_lookup(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"found": True, "tenant": t}
    return {"found": False}

def tenant_list(_):
    return {"tenants": TENANTS}

def tenant_is_owner(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"name": name, "is_owner": t["owner"]}
    return {"error": "tenant not found"}

def calculator(expr: str):
    if any(b in expr.lower() for b in BAN_PRHASES):
        return {"error": "blocked unsafe expression"}
    try:
        return {"result": eval(expr)}
    except Exception as e:
        return {"error": str(e)}
    

# dispatch table, normalized arg key diff between schema and func
TOOLS = {
    "calculator": lambda args: calculator(args.get("expr", "")),
    "tenant_lookup": lambda args: tenant_lookup(args.get("name", "")),
    "tenant_list": lambda _: tenant_list(),
    "tenant_is_owner": lambda args: tenant_is_owner(args.get("name", "")),
}

OLLAMA_TOOLS = [{
    "function_declarations": [
        {
            "name": "calculator",
            "description": "Safely evaluate math expressions.",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            }
        },
        {
            "name": "tenant_lookup",
            "description": "Look up a condo tenant by full name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        },
        {
            "name": "tenant_list",
            "description": "List all tenants.",
            "parameters": {
                "type": "object",
                "properties": {"dummy": {"type": "string"}},
                "required": []
            }
        },
        {
            "name": "tenant_is_owner",
            "description": "Check if a tenant is an owner.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        },
    ]
}]

class Memory:
    def __init__(self): self.data = {}
    def set(self,k,v): self.data[k] = v
    def get(self,k):  return self.data.get(k)


memory = Memory()

def llm(messages: list[str]) -> ChatResponse:
    response = chat(
        model=MODEL,
        messages=messages,
        tools=OLLAMA_TOOLS  
    )
    return response


def agent_loop(goal, max_steps=6):
    """
    Runs a ReAct-style loop:
      user message → model → tool_calls? → execute → append result → repeat
    until the model replies with plain text (no more tool calls).
    """
    history = memory.get("history") or [] 

    messages = [{
        "role": "system",
        "content": (
            f"""
                You are an agent. Your goal: {goal}
                Previous results: {history}
                Use the relevant tools: calculator, tenant_lookup, tenant_list, tenant_is_owner.
                If you want to use a tool, return JSON like:
                {{ "tool": "<tool_name>", "input": "<arg>" }}
                When done, return your final answer explicitly as: FINAL: <answer>
            """
            )
        },
        {"role": "user", "content": goal}
    ]
    response = llm(messages)
    msg = response.message

    # Persist the assistant turn in converstaion history
    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": msg.tool_calls or [],
    })

    if msg.tool_calls:
        #  Detect tool calls via the structured API field, not text heuristics
        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments or {}
            print(f"[tool] {name}({args})")

            result = TOOLS[name](args) if name in TOOLS else {"error": f"unknown tool: {name}"}
            print(f"[result] {result}")

            # Append tool result so the model can read it on the next turn
            messages.append({
                "role": "tool",
                "content": json.dumps(result),
            })
    else:
        # No tool calls → model is done reasoning, return its answer
        print(f"[answer] {msg.content}")
        return msg.content or ""
 
    return "Max steps reached without a final answer."

    

def plan(goal: str) -> list[str]:
    prompt = (
        "Break the following task into 2–4 clear, actionable steps. "
        "Return ONLY a JSON array of strings with no explanation or markdown.\n\n"
        f"Task: {goal}"
    )
    message = [{"role": "user", "content" : prompt}]
    response = llm(message)
    text = response.message.content.strip()
 
    # Strip ``` fences that some models add despite being told not to
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
 
    try:
        steps = json.loads(text)
        if isinstance(steps, list):
            return [str(s) for s in steps]
    except json.JSONDecodeError:
        pass
 
    # Fallback: one step per non-empty line
    return [line.strip() for line in text.splitlines() if line.strip()]

def reflect(step: str, answer: str) -> bool:
    prompt = (
        f"Step: {step}\n"
        f"Answer: {answer}\n\n"
        "Is this answer correct and complete for the step above? "
        "Reply with ONLY the word YES or NO."
    )
    message = [{"role": "user", "content" : prompt}]
    response = llm(message)

    verdict = response.message.content.strip().upper()
    passed = verdict.startswith("YES")
    print(f"  [reflect] {verdict} → {'pass' if passed else 'RETRY'}")
    return passed

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
 
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"\n  [retry {attempt}/{max_retries}]")
                # Give the model a hint about what went wrong
                step = f"{step}  (previous attempt gave wrong answer: {answer!r} — try again using tools)"
 
            answer = agent_loop(step)
 
            print(f"\n=== REVIEWER ===")
            if reflect(step, answer):
                break   # now stops retries when satisfied
 
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