import re
import json
import ast
from ollama import chat, ChatResponse

MODEL = "qwen3.5:9b"

TENANTS = [
    {"name": "Alice Johnson", "apartment": "12A", "owner": True},
    {"name": "Bob Smith",     "apartment": "8C",  "owner": False},
    {"name": "Carla Mendes",  "apartment": "15B", "owner": True},
]

# ---------------------------
# SAFE CALCULATOR
# ---------------------------
class SafeEval(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp,
        ast.Constant, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub
    )

    def visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f"Unsafe expression: {type(node).__name__}")
        return super().visit(node)

def safe_eval(expr: str):
    tree = ast.parse(expr, mode="eval")
    SafeEval().visit(tree)
    return eval(compile(tree, "", "eval"), {"__builtins__": {}})

# ---------------------------
# TOOLS
# ---------------------------
def tenant_list():
    return {"tenants": TENANTS}

def tenant_lookup(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"found": True, "tenant": t}
    return {"error": "tenant not found"}

def tenant_is_owner(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"name": name, "is_owner": t["owner"]}
    return {"error": "tenant not found"}

def tenant_apartment_number(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"found": True, "apartment": t["apartment"]}
    return {"error": "tenant not found"}

def calculator(expr: str):
    try:
        result = safe_eval(expr)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

TOOLS = {
    "calculator": calculator,
    "tenant_lookup": tenant_lookup,
    "tenant_is_owner": tenant_is_owner,
    "tenant_list": tenant_list,
    "tenant_apartment_number": tenant_apartment_number,
}

# ---------------------------
# TOOL PARSER
# ---------------------------
def parse_tool_call_from_text(text: str):
    if not text:
        return None, None

    text = re.sub(r"```(json)?", "", text).strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)

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

# ---------------------------
# LLM CALLS
# ---------------------------
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"]
            }
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
    }
]
def llm(messages) -> ChatResponse:
    return chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS)

def llm_plain(messages) -> ChatResponse:
    return chat(model=MODEL, messages=messages)

def safe_content(x):
    return x if isinstance(x, str) else json.dumps(x)

# ---------------------------
# MULTI-INTENT / DYNAMIC TOOL HELPERS
# ---------------------------
def extract_math_expr(goal: str):
    # Find anything that looks like a math expression
    match = re.search(r"[\d\.\+\-\*\/\(\)]+", goal)
    return match.group() if match else None

def extract_tenant_name(goal: str):
    goal_lower = goal.lower()
    for t in TENANTS:
        if t["name"].lower() in goal_lower:
            return t["name"]
    return None

def detect_intents(goal: str):
    """Detect which tools might be needed dynamically"""
    intents = []
    if any(w in goal.lower() for w in ["owner", "apartment", "tenant", "live"]):
        tenant_name = extract_tenant_name(goal)
        if tenant_name:
            if "owner" in goal.lower():
                intents.append(("tenant_is_owner", {"name": tenant_name}))
            elif "apartment" in goal.lower() or "live" in goal.lower():
                intents.append(("tenant_apartment_number", {"name": tenant_name}))
            else:
                intents.append(("tenant_lookup", {"name": tenant_name}))
    if "calculate" in goal.lower() or re.search(r"[\d\+\-\*\/\(\)]", goal):
        expr = extract_math_expr(goal)
        if expr:
            intents.append(("calculator", {"expr": expr}))
    return intents

def execute_multi_intent(goal: str):
    """Deterministically handle tenant + calculation intents."""
    results = []

    # Tenant ownership check
    if "owner" in goal.lower():
        name_str = extract_tenant_name(goal)
        if name_str:
            print(f"[forced tenant_is_owner] {name_str}")
            result = tenant_is_owner(name_str)
            if "error" not in result:
                results.append(
                    f"{name_str} is {'an owner' if result['is_owner'] else 'not an owner'}."
                )

    # Math calculation
    if "calculate" in goal.lower() or re.search(r"[\d\.\+\-\*\/\(\)]", goal):
        expr = extract_math_expr(goal)
        if expr:
            print(f"[forced calculator] {expr}")
            result = calculator(expr)
            if "result" in result:
                results.append(f"{expr} = {result['result']}")

    return " ".join(results) if results else None


def plan(goal: str):
    # For simplicity, each intent is one step
    intents = detect_intents(goal)
    steps = []
    for tool_name, args in intents:
        steps.append({"tool": tool_name, "args": args})
    # If nothing detected, treat the whole goal as one step
    if not steps:
        steps = [{"tool": None, "args": {}, "raw": goal}]
    return steps

def reflect(step: dict, result: str):
    # Simple deterministic reflection for calculation
    if step["tool"] == "calculator":
        return True
    if step["tool"] in ["tenant_is_owner", "tenant_apartment_number", "tenant_lookup"]:
        return "error" not in result
    return True

def execute_step(step: dict):
    tool = step.get("tool")
    args = step.get("args", {})
    raw = step.get("raw")

    # Tool call
    if tool:
        try:
            result = TOOLS[tool](**args)
        except Exception as e:
            result = {"error": str(e)}
        # Format human-readable answer
        if tool == "tenant_is_owner" and "error" not in result:
            return f"{result['name']} is {'an owner' if result['is_owner'] else 'not an owner'}."
        if tool == "tenant_apartment_number" and "error" not in result:
            return f"{args['name']} lives in apartment {result['apartment']}."
        if tool == "tenant_lookup" and "error" not in result:
            t = result["tenant"]
            return f"{t['name']} lives in {t['apartment']}, owner: {t['owner']}."
        if tool == "calculator" and "result" in result:
            return f"{args['expr']} = {result['result']}"
        return json.dumps(result)
    # Fallback: raw step text
    elif raw:
        return execute_multi_intent(raw)
# ---------------------------
# AGENT LOOP
# ---------------------------
def agent_loop(goal: str, max_steps: int = 6) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-using agent.\n"
                "You MUST respond with JSON tool calls.\n"
                "Never answer directly.\n"
                "Use tools for ALL tenant data and math."
            )
        },
        {"role": "user", "content": goal}
    ]

    for step in range(max_steps):
        print(f"\n--- step {step+1} ---")

        # Call LLM
        response = llm(messages)
        msg = response.message
        content = msg.content or ""
        messages.append({"role": "assistant", "content": safe_content(content)})

        # Try structured tool call from LLM
        name, args = parse_tool_call_from_text(content)
        if name:
            print(f"[tool] {name}({args})")
            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            print(f"[result] {result}")
            messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

            # Return result for known tools
            if name == "tenant_is_owner" and "error" not in result:
                return f"{result['name']} is {'an owner' if result['is_owner'] else 'not an owner'}."
            elif name == "calculator" and "result" in result:
                return f"{args['expr']} = {result['result']}"

        # If LLM didn’t produce valid tool call, use deterministic multi-intent executor
        fallback_result = execute_multi_intent(goal)
        if fallback_result:
            print(f"[final answer] {fallback_result}")
            return fallback_result

    return "Max steps reached"

def multi_agent(goal: str):
    print("\n=== PLAN ===")
    steps = plan(goal)
    for i, s in enumerate(steps, 1):
        print(f"{i}. {s}")

    results = []

    for step in steps:
        print(f"\n=== STEP: {step} ===")
        for attempt in range(3):
            answer = execute_step(step)
            if reflect(step, answer):
                break
            else:
                print(f"[retry {attempt+1}]")
        results.append(answer)

    final = " ".join(map(str, results))
    print("\n=== RESULTS ===")
    for s, r in zip(steps, results):
        print(f"{s} → {r}")

    return final

if __name__ == "__main__":
    # Try condo questions:
    # goal = "Is Bob Smith an owner?"
    # goal = "List every tenant in the building."
    # goal = "Which apartment does Alice Johnson live in?"
    # goal = "Check if Carla Mendes is an owner and calculate 500*1.08"
    # goal = "Is Carla Mendes an owner?"
    
    # goals = [
    #     "Is Carla Mendes an owner?",
    #     "Calculate 500*1.08",
    #     "Check if Carla Mendes is an owner and calculate 500*1.08",
    #     "List every tenant in the building."
    # ]

    # for g in goals:
    #     print("\n========================")
    #     print(f"Goal: {g}")
    #     print(agent_loop(g))

    goals = [
        "Is Bob Smith an owner?",
        "List every tenant in the building.",
        "Which apartment does Alice Johnson live in?",
        "Check if Carla Mendes is an owner and calculate 500*1.08",
        "Calculate 42 / 7 and check if Alice Johnson is an owner"
    ]

    for g in goals:
        print("\n========================")
        print(f"Goal: {g}")
        print(multi_agent(g))
