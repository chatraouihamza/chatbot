
# List of your JSON files to process
# json_files = ['devtest.json', 'test.json', 'val.json', 'val (1).json', 'devtest (1).json', 'test (1).json']

import json
import os

# Paths to the train.json and KB.json files
train_file = 'train_light.json'
kb_file = 'KB.json'

# Load the existing KB.json file
if os.path.exists(kb_file):
    with open(kb_file, 'r') as kb_input:
        kb_data = json.load(kb_input)
else:
    kb_data = {"intents": []}

# Load the train.json file
with open(train_file, 'r') as train_input:
    train_data = json.load(train_input)

# Process each entry in the train.json
for item in train_data:
    # Extract the main question
    main_question = item.get("question", "")
    item_id = item.get("id", "")
    tag = f"light_conver-{item_id}"

    # Extract all question-answer pairs from annotations
    patterns = [main_question]
    responses = []
    
    for annotation in item.get("annotations", []):
        if annotation["type"] == "multipleQAs":
            for qa_pair in annotation.get("qaPairs", []):
                patterns.append(qa_pair["question"])
                responses.extend(qa_pair["answer"])
        elif annotation["type"] == "singleAnswer":
            responses.extend(annotation.get("answer", []))

    # Create the new intent and add it to KB.json
    new_intent = {
        "tag": tag,
        "patterns": patterns,
        "responses": responses
    }

    kb_data['intents'].append(new_intent)

# Save the updated KB.json file
with open(kb_file, 'w') as kb_output:
    json.dump(kb_data, kb_output, indent=4)

print(f"Data from '{train_file}' has been merged into '{kb_file}' successfully!")
