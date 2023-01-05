from transformers import (
        BertTokenizerFast, 
    BertForMultipleChoice,
    BertForQuestionAnswering,
    pipeline
)
from argparse import ArgumentParser, Namespace
from pathlib import Path
import codecs
import json
import torch
import csv
import pandas as pd
from tqdm import tqdm

# Parse args
parser = ArgumentParser()

parser.add_argument(
    "--test_file",
    type=Path,
    default="./data/test_group.json",
)

parser.add_argument(
    "--ckpt_path",
    type=Path,
    default="./group/3",
)

parser.add_argument(
    "--output_path",
    type=Path,
    default="./pred_3.csv",
)

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

choices = []
 
subgroup = pd.read_csv("./data/subgroups.csv")
for item in subgroup["subgroup_name"]:
    choices.append(item)

# REF https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMultipleChoice
model = BertForMultipleChoice.from_pretrained(args.ckpt_path).to(device)
tokenizer = BertTokenizerFast.from_pretrained(args.ckpt_path)

with codecs.open(args.test_file, 'r', 'utf-8') as f:
    test_data = json.load(f)

predictions = []
cnt = 0

for data in tqdm(test_data):
    encoding = tokenizer(
        [data["sent1"]]*91, 
        choices, 
        return_tensors="pt", 
        padding=True,
        max_length=128,
        truncation=True
    )

    output = model(**{k: v.unsqueeze(0).to(device) for k, v in encoding.items()})

    # output.logits to see the logits

    # predictions.append([data["id"], predict_answer.replace(" ", "")])

    logits = output.logits
    answer = ""
    user_id = data["gold_source"]
    for _ in range(4):
        index = torch.argmax(logits)
        answer = answer + str(index.item()+1) + " "
        logits[0][index] = -1

    predictions.append([user_id, answer])



# Write to submission file
with open(args.output_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["user_id", "subgroup"])
    for row in predictions:
        writer.writerow(row)
