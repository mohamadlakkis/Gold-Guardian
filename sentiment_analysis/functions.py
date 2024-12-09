import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

SENTIMENT_ANALYSIS_MODEL = os.getenv("SENTIMENT_ANALYSIS_MODEL")


def get_sentiment_analysis(user_input):
    gpt_response = openai.chat.completions.create(
        model=SENTIMENT_ANALYSIS_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a Gold Assistant and News Analyzer that provides answers on whether the gold is a good investment given the news provided by the user. If no news were provided, your answer should be based on the current news that you are up to date with, you should also provide the news which your answer was based on. At the end of your response please add a new line providing if the investor should Invest (market is going up), Don't Invest (market is going down), or Not Sure (market is neutral). Write them in the following way:
                <br>
                <span style='color: green'>Invest</span>
                <span style='color: yellow'>Not Sure</span
                <span style='color: red'>Don't Invest</span>
                """,
            },
            {"role": "user", "content": user_input},
        ],
        max_tokens=3000,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return gpt_response.choices[0].message.content
