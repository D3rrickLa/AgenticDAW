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
        {"role": "user", "content": "Is Alice Johnson an owner?"},
        {
            "role": "assistant",
            "content": '{"name": "tenant_is_owner", "arguments": {"name": "Alice Johnson"}}'
        },
        {"role": "user", "content": goal}
    ]

    for step in range(max_steps):
        print(f"\n--- step {step+1} ---")

        response = llm(messages)
        msg = response.message

        content = msg.content or ""

        messages.append({
            "role": "assistant",
            "content": safe_content(content),
        })

        name, args = parse_tool_call_from_text(content)

        if name:
            print(f"[tool] {name}({args})")

            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = {"error": str(e)}

            print(f"[result] {result}")

            messages.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(result),
            })

            continue

        if "owner" in goal.lower():
            name_str = extract_tenant_name(goal)

            if name_str:
                print(f"[forced tenant_is_owner] {name_str}")

                result = tenant_is_owner(name_str)

                messages.append({
                    "role": "tool",
                    "name": "tenant_is_owner",
                    "content": json.dumps(result),
                })

            return f"{name_str} is {'an owner' if result['is_owner'] else 'not an owner'}."

        if "calculate" in goal.lower() or re.search(r"\d", goal):
            expr_match = re.search(r"[\d\.\+\-\*\/\(\)]+", goal)

            if expr_match:
                expr = expr_match.group()
                print(f"[forced calculator] {expr}")

                result = calculator(expr)

                messages.append({
                    "role": "tool",
                    "name": "calculator",
                    "content": json.dumps(result),
                })

                continue

    print(f"[final answer] {content}")
    return content

if __name__ == "__main__":
    # Try condo questions:
    # goal = "Is Bob Smith an owner?"
    # goal = "List every tenant in the building."
    # goal = "Which apartment does Alice Johnson live in?"
    # goal = "Check if Carla Mendes is an owner and calculate 500*1.08"
    goal = "Is Carla Mendes an owner?"
    print(agent_loop(goal))
