import pandas as pd
import torch
import codecs
import os
import json
import copy

train = pd.read_csv("./data/train_group.csv")
validation = pd.read_csv("./data/val_unseen_group.csv")
test = pd.read_csv("./data/test_unseen_group.csv")
subgroup = pd.read_csv("./data/subgroups.csv")
user_csv = pd.read_csv("./data/users.csv")

only_one = True

for i in range(train.shape[0]):
    if type(train["subgroup"][i]) != float:
        train["subgroup"][i] = train["subgroup"][i].split(' ')
    else:
        train["subgroup"][i] = []
    # print(type(train["subgroup"][i]))
dataset = {
        'train': train,
        'valid': validation,
        'test': test,
    }

# Use the user's interest as context
# Subgroups as choices 
# Preprocess it to the same format as swag dataset

# Build a dict of user id to info 
user = {}
for i in range(user_csv.shape[0]):
    entry = user_csv.iloc[i]
    user[entry["user_id"]] = {
            "gender": entry["gender"],
            "occupation_titles": entry["occupation_titles"],
            "interests": entry["interests"],
            "recreation_names": entry["recreation_names"],
        }

def preprocess_group(split):
# id, gold_source, sent1, sent2, ending0, label
    output = []
    index = 0
    for i in range(dataset[split].shape[0]):
        entry = dataset[split].iloc[i]
        user_info = user[entry["user_id"]]
        tmp = {}

        tmp["id"] = index
        index = index + 1

        tmp["gold_source"] = entry["user_id"]
        tmp["sent1"] = str(user_info["occupation_titles"]) + str(user_info["interests"])\
                + str(user_info["recreation_names"])
        tmp["sent1"] += " 男性" if user_info["gender"] == "male" else " 女性"
        tmp["sent2"] = "" # QUESTION Or should sent2 = sent1 ?

        for j in range(0, 91):
            key = "ending" + str(j)
            tmp[key] = subgroup.iloc[j]["subgroup_name"]

        if split != 'test':
            ans = entry["subgroup"]
            # print(ans)
            if type(ans) == float:
                tmp["label"] = int(l)-1
                output.append(copy.deepcopy(tmp))
                tmp["id"] = index
                index = index + 1
            else:
                for l in ans:
                    if l == '[' or l == ']' or l == '\'' or l == ' ':
                        continue
                    tmp["label"] = int(l)-1
                    output.append(copy.deepcopy(tmp))
                    tmp["id"] = index
                    index = index + 1
                    if only_one:
                        break
        else:
            output.append(copy.deepcopy(tmp))


    with codecs.open(os.path.join(f'./data/{split}_group_one.json'), 'w', 'utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

    return

if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:
        preprocess_group(split)

