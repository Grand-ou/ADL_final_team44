import numpy as np
import csv


def predict(result, path, user_ids, idx2user, domain='course'):
    output = []
    for i in range(len(result)):
        pred = ""
        for j in result[i]:
            pred += (str(j) + " ")
        pred = pred.strip()
            
        output.append([idx2user[user_ids[i]], pred])
    
    if domain == 'course':
            head =["user_id", "course_id"]
    else:
            head = ["user_id", "subgroup"]

    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(head)
        writer.writerows(output)


def predict_topic_from_course(result, user_ids, purch_hists, course2subgroup):
    result_boost = []
    for idx, i in enumerate(result):
        mix = purch_hists[user_ids[idx]] + list(i)
        result_boost.append(mix)
    
    pred = []
    for i in result_boost:
        weight = dict()
        for j in i:
            subgroup = course2subgroup[j]
            for k in subgroup:
                if k in weight:
                    weight[k] += 1
                else:
                    weight[k] = 1
        key  = list(weight.keys())
        val = list(weight.values())
        pred.append([
            key[j]
            for j in np.argsort(np.array(val))[::-1]
        ])
    
    return pred