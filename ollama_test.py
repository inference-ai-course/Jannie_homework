from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',  # Ollama server URL
    api_key='ollama',  # required but ignored by Ollama
)

response = client.chat.completions.create(
    model="llama2:13b",  # make sure this matches the model you pulled
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

print(response.choices[0].message.content)
