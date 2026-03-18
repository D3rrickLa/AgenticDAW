from ollama import chat, ChatResponse
import json
import re

MODEL = "qwen3.5:9b"
BAN_PHRASES = ["import", "os", "sys", "__", "eval", "exec"]

TENANTS = [
    {"name": "Alice Johnson", "apartment": "12A", "owner": True},
    {"name": "Bob Smith",     "apartment": "8C",  "owner": False},
    {"name": "Carla Mendes",  "apartment": "15B", "owner": True},
]

# TOOLS The Agentic has

def tenant_lookup(name: str):
    for t in TENANTS:
        if t["name"].lower() == name.lower():
            return {"found": True, "tenant": t}