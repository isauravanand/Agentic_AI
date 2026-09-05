import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# load_dotenv()
# my_api_key=os.getenv("GROQ_API_KEY")

# if not my_api_key:
#     raise ValueError("API key kaha hai bhai")

# client=Groq(api_key=my_api_key)
# model = "openai/gpt-oss-120b"

# 1. Read file
with open("document.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. Extract text
# print("Original text:")
# print(text)


# 3. Split into chunks
chunk_size = 200
overlap = 50

chunks = []

start = 0

while start < len(text):

    # Take 200 characters
    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    # 4. Use overlap
    start = end - overlap


# 5. Give every chunk an ID
for i, chunk in enumerate(chunks):

    chunk_id = f"chunk_{i + 1}"

    print("\n----------------------")
    print("ID:", chunk_id)
    print("Text:")
    print(chunk)