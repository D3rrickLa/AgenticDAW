from ollama import chat

'''
Agent Loop order (LangChain)
LLM Reasoning -> Call tool -> Observe result -> store in mem -> repeat
'''


messages = [
    {
        "role" : "user",
        "content" : "Tell me a short story."
    }
]

response = chat(model="qwen2.5-coder:3b", messages=messages, stream=True)
for chunk in response:
    print(chunk.message.content, end="", flush=True)