import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

base={
     "age":"saurav anand age is 21",
        "net worth":"His net worth is 10 rs"
}

def information(question):
    question=question.lower()
    if "age" in question:
        return base["age"]
    elif "net worth" in question:
        return base["net worth"]
    else:
        return "Sorry"  


def ask_llm(question):
    context=information(question)
    sys_prompt=f""" answer only in 1 line according to this context , do not halicunate , context={context}"""
    sys_msg={
        "role":"system",
        "content":sys_prompt
    }
    message={
        "role":"user",
        "content":question
    }
    messages=[sys_msg,message]
    response=client.chat.completions.create(model=model,messages=messages)
    answer=response.choices[0].message.content
    return answer

question="do you know saurav anand net worth "
print(ask_llm(question))