import openai
openai.api_key = "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"

gpt_response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:question-answer-v3:AaodkYTJ:ckpt-step-382",
            messages=[
                {"role": "system", "content": "You are a Gold Assistant that provides accurate answers to questions about gold."},
                {"role": "user", "content": "How do you determine how much you should put into gold versus how much you should put to work?"}
            ], 
            max_tokens = 3000,
            temperature = 0,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
final_answer = gpt_response["choices"][0]["message"]["content"]
print(final_answer)