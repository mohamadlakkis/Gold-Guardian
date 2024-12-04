import json
import random
# File paths
input_file = "Q_AND_A_TRAIN_3.jsonl"
output_file = "output.jsonl"

# # Updated system content
# new_system_content = "You are a Gold Assistant that provides accurate answers to questions about gold."

# # Read, update, and write the JSONL file
# with open(input_file, "r") as infile, open(output_file, "w") as outfile:
#     for line in infile:
#         # Parse the JSON object
#         data = json.loads(line)
#         # Update the system content
#         for message in data.get("messages", []):
#             if message.get("role") == "system":
#                 message["content"] = new_system_content
#         # Write the updated JSON object back to the new file
#         json.dump(data, outfile)
#         outfile.write("\n")

# print(f"Updated JSONL file saved as {output_file}.")
input_file = "output.jsonl"
output_10_file = "split_10.jsonl"
output_90_file = "split_90.jsonl"

# Read all lines from the JSONL file
with open(input_file, "r") as infile:
    lines = infile.readlines()

# Shuffle the lines randomly
random.shuffle(lines)

# Calculate the split index
split_index = int(len(lines) * 0.1)

# Split the data into 10% and 90%
lines_10 = lines[:split_index]
lines_90 = lines[split_index:]

# Write 10% of the data to a new file
with open(output_10_file, "w") as outfile_10:
    outfile_10.writelines(lines_10)

# Write 90% of the data to another file
with open(output_90_file, "w") as outfile_90:
    outfile_90.writelines(lines_90)