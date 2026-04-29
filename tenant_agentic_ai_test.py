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
    },
    {
    "type": "function",
        "function": {
            "name": "tenant_apartment_number",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
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
    try:
        data = json.loads(text)
        name = data.get("name")
        args = data.get("arguments", {})
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

# ---------------------------
# AGENTIC AI: LLM-in-the-loop
# ---------------------------
def agentic_llm(goal: str, max_steps=8):
    messages = [
        {
            "role": "system",
            "content": f"""
        You are an autonomous agent that can use tools.

        Available tenants:
        {TENANTS}

        Guidelines:
        - Think step-by-step (you may explain briefly)
        - Use tools when needed
        - After each tool result, decide what to do next
        - When done, give a final answer

        When calling a tool, use the tool system (not text).
        Do not return empty responses.
        """
        },
        {"role": "user", "content": goal}
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")

        response = llm(messages)
        msg = response.message

        content = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None)

        # ---------------------------
        # PRINT REASONING (LIGHT)
        # ---------------------------
        if content.strip():
            print(f"[THINK] {content}")

        # ---------------------------
        # TOOL CALLS
        # ---------------------------
        if tool_calls:
            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments", {})

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                print(f"[tool call] {name}({args})")

                if name not in TOOLS:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = TOOLS[name](**args)
                    except Exception as e:
                        result = {"error": str(e)}

                print(f"[tool result] {result}")

                messages.append({
                    "role": "assistant",
                    "tool_calls": [call]
                })

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result)
                })

                # ---------------------------
                # REFLECTION PROMPT (SOFT)
                # ---------------------------
                messages.append({
                    "role": "user",
                    "content": f"""
                    Tool result: {json.dumps(result)}

                    What did you learn from this?
                    What should you do next?
                    """
                })

            continue

        # ---------------------------
        # FINAL ANSWER
        # ---------------------------
        if content.strip():
            print("\n=== FINAL ANSWER ===")
            print(content)
            return content

        # ---------------------------
        # FAILSAFE
        # ---------------------------
        print("[warning] empty response, nudging model")

        messages.append({
            "role": "user",
            "content": "Please continue reasoning or give the final answer."
        })

    return "Failed to complete task."
# ---------------------------
# TEST
# ---------------------------
if __name__ == "__main__":
    goals = [
        "Check if Carla Mendes is an owner and calculate 500*1.08",
        # "Calculate 42 / 7 and check if Alice Johnson is an owner",
        "List every tenant in the building."
    ]
    for g in goals:
        agentic_llm(g)