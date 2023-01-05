import numpy as np

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


def course_metrics(model, samples, user_item_data, course2idx, user2idx):
    preds = []
    actuals = []
    for i in range(samples.shape[0]):
        sample = samples.iloc[i]
        userid = user2idx[sample['user_id']]
        pred, score = model.recommend(userid, user_item_data[userid], N=50)
        preds.append(pred)
        actuals.append([course2idx[i] for i in sample['course_id'].split(' ')])
    return mapk(actuals, preds)


def group_metrics(model, samples, user_item_data, user2idx, course2subgroup):
    preds = []
    actuals = []
    for i in range(samples.shape[0]):
        sample = samples.iloc[i]
        userid = user2idx[sample['user_id']]
        pred, score = model.recommend(userid, user_item_data[userid], N=50)
        subgroupids = [course2subgroup[j] for j in pred]
        subgroupids = [k for j in subgroupids for k in j]  # flatten: list of list -> list
        
        preds.append(list(set(subgroupids)))
        try:
            actuals.append(sample['subgroup'].split(' '))
        except:  # nan
            actuals.append([0])
    return mapk(actuals, preds)