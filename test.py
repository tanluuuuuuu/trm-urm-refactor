from openai import OpenAI

client = OpenAI(
    api_key="ab9ce166c0fa4433ac564a04e9d6ec60.pX5Evk9iCOLCvnkI",
    base_url="https://api.z.ai/api/paas/v4/"  # ← Changed from /coding/paas/v4/
)

resp = client.chat.completions.create(
    model="glm-4.7",
    messages=[{"role": "user", "content": "Test GLM-4.7 API"}]
)
print(resp.choices[0].message.content)
