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

def llm(prompt, use_tools=False) -> ChatResponse:
    response = chat(
        model=MODEL,
        messages=[
            {
                {
                "role": "system",
                "content": (
                    "You are a helpful agent with access to tools. "
                    "Always use the provided tools to look up real data rather than guessing. "
                    "When you have a definitive answer, state it clearly in plain text."
                )
            },
            },
            {"role": "user", "content": prompt}
        ],
        tools=OLLAMA_TOOLS if use_tools else None,
        stream=True
    )
    return response


def agent_loop(goal, max_steps=6):
    """
    Runs a ReAct-style loop:
      user message → model → tool_calls? → execute → append result → repeat
    until the model replies with plain text (no more tool calls).
    """
    history = memory.get("history") or []

    for step in range(max_steps):
        print(f"\n--- STEP {step + 1} ---")

        # Include history in prompt so the model can reason over previous results
        response = llm(
            f"""
            You are an agent. Your goal: {goal}
            Previous results: {history}

            Use the relevant tools: calculator, tenant_lookup, tenant_list, tenant_is_owner.
            Return your final answer explicitly as: "FINAL: <answer>"
            """
        )

        # if "tool" in text:
        #     try:
        #         data = json.loads(text)
        #     except:
        #         print("Model did not return JSON:", text)
        #         continue

        #     tool_name = data["tool"]
        #     arg = data.get("input")
        #     result = TOOLS[tool_name](arg)

        #     history.append(result)
        #     memory.set("history", history)

        #     continue

        # if text.startswith("FINAL:"):
        #     return text


def plan(goal):
    p = llm(f"Break this task into 3 short steps:\n{goal}")
    print(p)
    return p

def reflect(answer, context=None):
    prompt = f"""
        Step: {context}
        Answer: {answer}
        Is this answer correct for the step? Respond ONLY with YES or NO.
        """
    r = llm(prompt)
    print("Reflection:", r)
    return r

def multi_agent(goal):
    print("\n=== PLANNING ===")
    steps_text = plan(goal)
    steps = [s for s in steps_text.split("\n") if s.strip() and s.strip()[0] in "123"]

    results = []
    last_result = None
    for step in steps:
        print(f"\n=== WORKER executing: {step} ===")
        result = agent_loop(step)
        last_result = result
        results.append(result)

        print("\n=== REVIEWER ===")
        reflect(result, context=step)

    print("\n=== FINAL RESULTS ===")
    return results

if __name__ == "__main__":
    # Try condo questions:
    # goal = "Is Bob Smith an owner?"
    # goal = "List every tenant in the building."
    # goal = "Which apartment does Alice Johnson live in?"

    goal = "Check if Carla Mendes is an owner and calculate 500*1.08"
    print(multi_agent(goal))