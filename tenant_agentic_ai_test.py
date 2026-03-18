from ollama import chat, ChatResponse
import json
import re

MODEL = "qwen3.5:4b"
BAN_PHRASES = ["import", "os", "sys", "__", "eval", "exec"]

TENANTS = [
    {"name": "Alice Johnson", "apartment": "12A", "owner": True},
    {"name": "Bob Smith",     "apartment": "8C",  "owner": False},
    {"name": "Carla Mendes",  "apartment": "15B", "owner": True},
]

# TOOLS The Agentic has
def tenant_list():
    return {"tenants": TENANTS}

def tenant_lookup(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"found": True, "tenant": t}
    return {"found": False}

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
    
TOOLS = {
    "calculator":      lambda args: calculator(args.get("expr", "")),
    "tenant_list":     lambda args: tenant_list(),
    "tenant_lookup":   lambda args: tenant_lookup(args.get("name", "")),
    "tenant_is_owner": lambda args: tenant_is_owner(args.get("name", "")),
}

OLLAMA_TOOLS = [
        {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string"}
                },
                "required": ["expr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_lookup",
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
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_is_owner",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        }
    }
]

def parse_tool_call_from_text(text: str):
    if not text:
        return None, None
    
    text = re.sub(r"```(json)?", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, None
    
    try:
        data = json.loads(match.group())
        name = (
            data.get("name")
                or data.get("tool")
                or data.get("function", {}).get("name")
        )

        args = (
            data.get("arguments")
                or data.get("parameters")
                or data.get("args")
                or {}
        )

        if isinstance(args, str):
            args = json.loads(args)

        if name in TOOLS:
            return name, args
        
    except Exception:
        pass 

    return None, None

def llm(messages) -> ChatResponse:
    return chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS)

def llm_plain(messages) -> ChatResponse:
    return chat(model=MODEL, messages=messages)

def safe_content(x):
    if isinstance(x, str):
        return x
    return json.dumps(x)

def agent_loop(goal: str, max_steps: int = 6) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-using agent.\n"
                "You MUST use tools for calculations and data.\n"
                "DO NOT guess.\n"
                "Only give final answer after tools are used."
            )
        },
        {"role": "user", "content": goal}
    ]

    for step in range(max_steps):
        print(f"\n--- step {step+1} ---")

        response = llm(messages)
        msg = response.message

        messages.append({
            "role": "assistant",
            "content": safe_content(msg.content or ""),
            "tool_calls": msg.tool_calls or "",
        })

        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or {}

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                print(f"[tool] {name}({args})")

                result = TOOLS.get(name, lambda _: {"error": "unknown"})(args)
                print(f"[result] {result}")

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result),
                })

            continue

        if msg.content:
            name, args = parse_tool_call_from_text(msg.content)

            if name:
                print(f"[fallback tool] {name}({args})")

                result = TOOLS[name](args)
                print(f"[result] {result}")

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result),
                })

                continue

        print(f"[answer] {msg.content}")
        return safe_content(msg.content or "")

    return "Max steps reached"

def plan(goal: str):
    prompt = (
        "Break into 2–4 steps. Return JSON array only.\n\n"
        f"Task: {goal}"
    )

    res = llm_plain([{"role": "user", "content": prompt}])
    text = res.message.content.strip()

    text = re.sub(r"```(json)?", "", text).strip()

    try:
        return json.loads(text)
    except:
        return [goal]
    
def reflect(step: str, answer: str):
    if "calculate" in step.lower():
        return True  # deterministic

    prompt = (
        f"Step: {step}\nAnswer: {answer}\n\n"
        "Is this strictly correct? YES or NO only."
    )

    res = llm_plain([{"role": "user", "content": prompt}])
    verdict = res.message.content.strip().upper()

    print(f"[reflect] {verdict}")
    return verdict.startswith("YES")

def multi_agent(goal: str):
    print("\n=== PLAN ===")
    steps = plan(goal)

    for i, s in enumerate(steps, 1):
        print(f"{i}. {s}")

    results = []

    for step in steps:
        print(f"\n=== STEP: {step} ===")

        answer = None

        for attempt in range(3):
            answer = agent_loop(step)

            if reflect(step, answer):
                break
            else:
                print(f"[retry {attempt+1}]")

        results.append(answer)

    print("\n=== RESULTS ===")
    for s, r in zip(steps, results):
        print(f"{s} → {r}")

    return results

if __name__ == "__main__":
    # Try condo questions:
    # goal = "Is Bob Smith an owner?"
    # goal = "List every tenant in the building."
    # goal = "Which apartment does Alice Johnson live in?"
    goal = "Check if Carla Mendes is an owner and calculate 500*1.08"
    multi_agent(goal)