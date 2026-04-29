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
# LLM CALL
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
    },
    {
        "type": "function",
        "function": {
            "name": "tenant_list",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def llm(messages) -> ChatResponse:
    return chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS)

def safe_content(x):
    return x if isinstance(x, str) else json.dumps(x)

# ---------------------------
# TOOL PARSER
# ---------------------------
def parse_tool_call_from_text(text: str):
    text = re.sub(r"```(json)?", "", text or "").strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None, None
    try:
        data = json.loads(match.group())
        name = data.get("name") or data.get("tool") or data.get("function", {}).get("name")
        args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
        if isinstance(args, str):
            args = json.loads(args)
        if name in TOOLS:
            return name, args
    except Exception:
        pass
    return None, None

def extract_math_expr(goal: str):
    matches = re.findall(r"[\d\.\+\-\*\/\(\)]+", goal)
    return max(matches, key=len) if matches else None

def extract_tenant_name(goal: str):
    goal_lower = goal.lower()
    for t in TENANTS:
        if t["name"].lower() in goal_lower:
            return t["name"]
    return None

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
    if "calculate" in goal.lower() or re.search(r"\d", goal):
        expr = extract_math_expr(goal)
        if expr:
            print(f"[forced calculator] {expr}")
            result = calculator(expr)
            if "result" in result:
                results.append(f"{expr} = {result['result']}")

    return " ".join(results) if results else None
# ---------------------------
# AGENTIC AI: LLM-in-the-loop
# ---------------------------
def agentic_llm(goal: str, max_steps=6):
    messages = [
        {"role": "system", "content": "You are a tool-using agent. Plan, reflect, and call tools to achieve the user's goal."},
        {"role": "user", "content": goal}
    ]

    results = []

    for step_num in range(max_steps):
        print(f"\n--- Step {step_num+1} ---")
        response = llm(messages)
        msg = response.message
        content = msg.content or ""
        messages.append({"role": "assistant", "content": safe_content(content)})

        # Try tool call
        name, args = parse_tool_call_from_text(content)
        if name:
            print(f"[tool call] {name}({args})")
            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            print(f"[tool result] {result}")
            messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

            # Reflect: ask LLM if result is enough to continue or retry
            messages.append({"role": "user", "content": f"Did this tool call succeed for the goal? Result: {json.dumps(result)}"})
            continue
        if not name:
            fallback_result = execute_multi_intent(goal)
            if fallback_result:
                results.append(fallback_result)
                break
        if "list" in goal.lower() and "tenant" in goal.lower():
            print("[forced tenant_list]")
            result = tenant_list()
            return ", ".join([t["name"] for t in result["tenants"]])
        
        # Fallback: no tool, maybe LLM reasoning required
        if content.strip():
            print(f"[LLM reasoning] {content}")
            results.append(content)

    final_answer = " ".join(results)
    print("\n=== FINAL AGENTIC ANSWER ===")
    print(final_answer)
    return final_answer

# ---------------------------
# TEST
# ---------------------------
if __name__ == "__main__":
    goals = [
        "Check if Carla Mendes is an owner and calculate 500*1.08",
        "Calculate 42 / 7 and check if Alice Johnson is an owner",
        "List every tenant in the building."
    ]
    for g in goals:
        agentic_llm(g)