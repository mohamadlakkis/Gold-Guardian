import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

Q_AND_A_MODEL = os.getenv("Q_AND_A_MODEL")


def answer_q_and_a(user_input, old_conversation=None):
    messages = [
        {
            "role": "system",
            "content": "You are a Gold Assistant that provides accurate answers to questions about gold. You are also a chatbot that can answer questions about gold, and take a conversational approach to answering questions. Include the date of your response. DO NOT MENTION THE PRICE OF GOLD, EVEN IF ASKED.",
        }
    ]

    if old_conversation:
        messages.extend(old_conversation[-10:])

    messages.append({"role": "user", "content": user_input})

    gpt_response = openai.chat.completions.create(
        model=Q_AND_A_MODEL,
        messages=messages,
        max_tokens=3000,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    messages.append(
        {
            "role": gpt_response.choices[0].message.role,
            "content": gpt_response.choices[0].message.content,
        }
    )

    return gpt_response.choices[0].message.content
