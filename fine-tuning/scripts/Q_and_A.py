
import pandas as pd
import json
# Load the dataset
file_path = "fine-tuning/Q_AND_A.csv"  # Update with your file path
df = pd.read_csv(file_path)


system_message = "You are a Gold Sentiment Analysis assistant that provides accurate sentiment predictions."

# Prepare JSONL file
output_file = "Q_AND_A_TRAIN.jsonl"
with open(output_file, "w") as f:
    for _, row in df.iterrows():
        conversation = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": row["Question"]},
                {"role": "assistant", "content": row["Answer"]}
            ]
        }
        f.write(json.dumps(conversation) + "\n")

print(f"Dataset has been converted and saved to {output_file}")
