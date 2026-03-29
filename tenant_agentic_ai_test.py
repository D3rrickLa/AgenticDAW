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
# LLM CALLS
# ---------------------------
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_is_owner",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_lookup",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        }
    }
]

def llm(messages) -> ChatResponse:
    return chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS)

def safe_content(x):
    return x if isinstance(x, str) else json.dumps(x)

# ---------------------------
# AGENTIC AI HELPERS
# ---------------------------
def extract_math_expr(text: str):
    match = re.search(r"[\d\.\+\-\*\/\(\)]+", text)
    return match.group() if match else None

def extract_tenant_name(text: str):
    text_lower = text.lower()
    for t in TENANTS:
        if t["name"].lower() in text_lower:
            return t["name"]
    return None

def detect_intents(goal: str):
    """Ask LLM to propose tool steps or fallback to deterministic detection."""
    intents = []
    tenant_name = extract_tenant_name(goal)

    if tenant_name:
        if "owner" in goal.lower():
            intents.append(("tenant_is_owner", {"name": tenant_name}))
        elif "apartment" in goal.lower() or "live" in goal.lower():
            intents.append(("tenant_apartment_number", {"name": tenant_name}))
        else:
            intents.append(("tenant_lookup", {"name": tenant_name}))

    expr = extract_math_expr(goal)
    if expr:
        intents.append(("calculator", {"expr": expr}))

    return intents

def reflect(step, result):
    """Decide if we should accept result or retry."""
    if step["tool"] in ["tenant_is_owner", "tenant_apartment_number", "tenant_lookup"]:
        return "error" not in result
    if step["tool"] == "calculator":
        return "result" in result
    return True

def execute_step(step):
    tool = step.get("tool")
    args = step.get("args", {})
    if tool:
        result = TOOLS[tool](**args)
        # Human-readable formatting
        if tool == "tenant_is_owner" and "error" not in result:
            return f"{result['name']} is {'an owner' if result['is_owner'] else 'not an owner'}."
        if tool == "tenant_apartment_number" and "error" not in result:
            return f"{args['name']} lives in apartment {result['apartment']}."
        if tool == "calculator" and "result" in result:
            return f"{args['expr']} = {result['result']}"
        return json.dumps(result)
    return None

def plan(goal: str):
    steps = []
    intents = detect_intents(goal)
    for t, a in intents:
        steps.append({"tool": t, "args": a})
    if not steps:
        steps = [{"tool": None, "raw": goal}]
    return steps

def agentic_loop(goal: str, max_steps=6):
    print(f"\n=== Goal: {goal} ===")
    steps = plan(goal)
    results = []

    for step in steps:
        print(f"\n--- Step: {step} ---")
        for attempt in range(3):
            answer = execute_step(step)
            if reflect(step, answer):
                break
            else:
                print(f"[Retry {attempt+1}]")
        results.append(answer)

    final = " ".join(results)
    print("\n=== Final Answer ===")
    print(final)
    return final

# ---------------------------
# TEST GOALS
# ---------------------------
if __name__ == "__main__":
    goals = [
        "Is Bob Smith an owner?",
        "List every tenant in the building.",
        "Which apartment does Alice Johnson live in?",
        "Check if Carla Mendes is an owner and calculate 500*1.08",
        "Calculate 42 / 7 and check if Alice Johnson is an owner"
    ]

    for g in goals:
        agentic_loop(g)