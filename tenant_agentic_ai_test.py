import re
import json
import ast
from ollama import chat, ChatResponse

MODEL = "qwen3.5:4b"

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

def extract_tenant_name(goal: str):
    goal_lower = goal.lower()

    for t in TENANTS:
        if t["name"].lower() in goal_lower:
            return t["name"]

    return None

def extract_math_expr(goal: str):
    match = re.search(r"[\d\.\+\-\*\/\(\)]+", goal)
    return match.group() if match else None


def extract_tenant_name(goal: str):
    goal_lower = goal.lower()

    for t in TENANTS:
        if t["name"].lower() in goal_lower:
            return t["name"]

    return None

# ---------------------------
# AGENT LOOP
# ---------------------------
def agent_loop(goal: str) -> str:
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

    results = []

    # ---------------------------
    # Try LLM tool calls first
    # ---------------------------
    response = llm(messages)
    msg = response.message
    content = msg.content or ""
    messages.append({"role": "assistant", "content": safe_content(content)})

    # If LLM returned a structured tool call
    name, args = parse_tool_call_from_text(content)
    if name:
        print(f"[tool] {name}({args})")
        try:
            result = TOOLS[name](**args)
        except Exception as e:
            result = {"error": str(e)}
        print(f"[result] {result}")
        messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

        # Collect LLM tool call result
        if name == "tenant_is_owner" and "error" not in result:
            results.append(
                f"{result['name']} is {'an owner' if result['is_owner'] else 'not an owner'}."
            )
        elif name == "calculator" and "result" in result:
            results.append(f"{args['expr']} = {result['result']}")

    # ---------------------------
    # Multi-intent fallback (deterministic)
    # ---------------------------

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

    # ---------------------------
    # Return combined results
    # ---------------------------
    if results:
        final = " ".join(results)
        print(f"[final answer] {final}")
        return final

    return "No valid tool action detected."

if __name__ == "__main__":
    # Try condo questions:
    # goal = "Is Bob Smith an owner?"
    # goal = "List every tenant in the building."
    # goal = "Which apartment does Alice Johnson live in?"
    goal = "Check if Carla Mendes is an owner and calculate 500*1.08"
    # goal = "Is Carla Mendes an owner?"
    print(agent_loop(goal))
