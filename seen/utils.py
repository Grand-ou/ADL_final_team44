import numpy as np
import csv



def apk(actual, predicted, k=50):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=50):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


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


def knn_course_predict(indices, purch_hists, user_ids):
    output = []
    for i in user_ids:
        purchased = purch_hists[i]
        weight = dict()
        for j in indices[i][1:]:
            for k in purch_hists[j]:
                if k not in purchased and k in weight:
                        weight[k] += 1
                elif k not in purchased and k not in weight:
                        weight[k] = 1
                else:
                    continue
    
        key  = list(weight.keys())
        val = list(weight.values())
        output.append([
            key[j]
            for j in np.argsort(np.array(val))[::-1]
        ])

    return output

def mix1_course_predict(base, mix1):
    pred_mix1 = []
    for (idx, i) in enumerate(base):
        weight = dict()
        for idx_j, j in enumerate(i):
            weight[j] = idx_j
            if j in mix1[idx]:
                weight[j] += np.where(mix1[idx] == j)[0][0]
            else:
                weight[j] += len(mix1[idx])
        
        key  = list(weight.keys())
        val = list(weight.values())
        pred_mix1.append([
            key[j]
            for j in np.argsort(np.array(val))
        ])

    return pred_mix1


def mix2_course_predict(base, mix1, mix2):
    pred_mix2 = []
    for (idx, i) in enumerate(base):
        weight = dict()
        for idx_j, j in enumerate(i):
            weight[j] = idx_j
            if j in mix1[idx] and j in mix2[idx]:
                weight[j] += (np.where(mix1[idx] == j)[0][0] + np.where(mix2[idx] == j)[0][0])
            elif j in mix1[idx]:
                weight[j] += (np.where(mix1[idx] == j)[0][0] + len(mix2[idx]))
            elif j in mix2[idx]:
                weight[j] += (len(mix1[idx]) + np.where(mix2[idx] == j)[0][0])
            else:
                weight[j] += (len(mix1[idx]) + len(mix2[idx]))
        
        key  = list(weight.keys())
        val = list(weight.values())
        pred_mix2.append([
            key[j]
            for j in np.argsort(np.array(val))
        ])

    return pred_mix2

