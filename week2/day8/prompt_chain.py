import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_llm(prompt):

    response = client.chat.completions.create(
        model="groq/compound-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


review = """
The food was amazing and the staff were friendly,
but my order took almost an hour to arrive.
The restaurant was also quite expensive.
"""


# -------------------------
# CHAIN 1
# -------------------------

analysis = ask_llm(
    f"""
Analyze this customer review and extract the
important points.

Review:
{review}
"""
)

print("\nCHAIN 1 - ANALYSIS")
print(analysis)


# -------------------------
# CHAIN 2
# -------------------------

sentiment = ask_llm(
    f"""
Determine the overall sentiment of this customer review.

Analysis:
{analysis}

Return one of:
Positive
Negative
Mixed
"""
)

print("\nCHAIN 2 - SENTIMENT")
print(sentiment)


# -------------------------
# CHAIN 3
# -------------------------

problems = ask_llm(
    f"""
Identify the main problems mentioned in this review.

Review:
{review}

Analysis:
{analysis}

Sentiment:
{sentiment}
"""
)

print("\nCHAIN 3 - PROBLEMS")
print(problems)


# -------------------------
# CHAIN 4
# -------------------------

final_response = ask_llm(
    f"""
Write a professional response to the customer.

Original Review:
{review}

Analysis:
{analysis}

Sentiment:
{sentiment}

Problems:
{problems}

The response should:
- Thank the customer
- Acknowledge the positive feedback
- Apologize for the problems
- Sound professional
"""
)

print("\nCHAIN 4 - FINAL RESPONSE")
print(final_response)