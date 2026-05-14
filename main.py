from ollama import chat

'''
Agent Loop order (LangChain)
LLM Reasoning -> Call tool -> Observe result -> store in mem -> repeat
'''
MODEL = "qwen3.5:9b"

messages = [
    {
        "role" : "user",
        "content" : "Tell me a short story."
    }
]

response = chat(model=MODEL, messages=messages, stream=True)
for chunk in response:
    print(chunk.message.content, end="", flush=True)