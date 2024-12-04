# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("ankurzing/sentiment-analysis-in-commodity-market-gold")

# print("Path to dataset files:", path)
import pandas as pd
import json
file_path = "fine-tuning/gold-dataset-sinha-khandait.csv"  
df = pd.read_csv(file_path)

system_message = "You are a Gold Sentiment Analysis assistant that provides accurate sentiment predictions."

output_file = "gold_sentiment_analysis.jsonl"
with open(output_file, "w") as f:
    for _, row in df.iterrows():
        conversation = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": row["News"]},
                {"role": "assistant", "content": row["Price Sentiment"]}
            ]
        }
        f.write(json.dumps(conversation) + "\n")

print(f"Dataset has been converted and saved to {output_file}")
