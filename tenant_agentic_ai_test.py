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
    if any(b in expr.lower() for b in BAN_PHRASES):
        return {"error": "blocked unsafe expression"}
    try:
        return {"result": eval(expr)}
    except Exception as e:
        return {"error": str(e)}

TOOLs = {
    "calculator": calculator,
    "tenant_lookup": tenant_lookup,
    "tenant_is_owner": tenant_is_owner,
    "tenant_list": tenant_list
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
            "name": "tenant_apartment_number",
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

if __name__ == "__main__":
    # Try condo questions:
    # goal = "Is Bob Smith an owner?"
    # goal = "List every tenant in the building."
    # goal = "Which apartment does Alice Johnson live in?"
    goal = "Check if Carla Mendes is an owner and calculate 500*1.08"
