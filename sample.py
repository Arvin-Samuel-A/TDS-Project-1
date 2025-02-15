import openai

openai.api_key = ""

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response["choices"][0]["message"]["content"])
