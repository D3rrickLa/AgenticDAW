from ollama import chat

messages = [
    {
        "role" : "user",
        "content" : "Tell me a short story."
    }
]

response = chat(model="qwen2.5-coder:3b", messages=messages, stream=True)
for chunk in response:
    print(chunk.message.content, end="", flush=True)