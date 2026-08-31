import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def stream_response(prompt):

    response = client.chat.completions.create(
           model="groq/compound-mini",


        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.7,

        stream=True
    )

    for chunk in response:

        text = chunk.choices[0].delta.content or ""

        print(text, end="", flush=True)


print("🤖 AI: ")

stream_response(
    "Explain Prompt Chaining in simple words."
)

print("\n\n✅ Done")