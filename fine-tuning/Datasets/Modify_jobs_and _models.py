from openai import OpenAI

client = OpenAI(api_key="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")

client.models.delete("ft:gpt-4o-mini-2024-07-18:personal:q-and-a:AajU3xoJ")