import os
import csv
import argparse
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MultiLabelBinarizer

from ckiptagger import WS
def apk(actual, predicted, k=10):
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

def mapk(actual, predicted, k=10):
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
parser = argparse.ArgumentParser()
parser.add_argument("smooth_method", default='add_one', help="smoothing method")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ws = WS("./data", disable_cuda=False)
# data_utils.download_data_gdown("./")

data_path = 'hahow/data/'
courses = pd.read_csv(data_path+'courses.csv')
users = pd.read_csv(data_path+'users.csv')
train = pd.read_csv(data_path+'train.csv')
valid = pd.read_csv(data_path+'val_unseen.csv')
# test = pd.read_csv(data_path+'test_unseen.csv')
users['gender'] = users['gender'].map({'male': '男', 'female': '女', 'other': '三性'})
valid = valid.merge(users, how='left', on='user_id')
# test = test.merge(users, how='left', on='user_id')
train = train.merge(users, how='left', on='user_id')
def query_genarate(record):
    start = time.time()
    record = record.fillna('')
    record['query'] = record['gender']+' '+record['occupation_titles']+' '+record['interests']+' '+record['recreation_names']
    record['query'] = record['query'].str.replace('其他', ' ')
    record['query'] = record['query'].str.replace(',', ' ')
    record['query'] = record['query'].str.replace('_', ' ')
    record['query_list'] = ws(record['query'].tolist())
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    return record
def list_value_to_column(record, col='query_list' ):
    
    mlb = MultiLabelBinarizer(sparse_output=True)

    return record.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(record.pop(col)),
                index=record.index,
                columns=mlb.classes_))
# test = query_genarate(test)
valid = query_genarate(valid)
train = query_genarate(train)
query_mat = pd.DataFrame()
query_mat['query_list'] = train['query_list']
query_mat = list_value_to_column(query_mat)
query_mat.drop([' ', '  ', '   ', '、'], axis=1, inplace = True)
course_mat = pd.DataFrame()
course_mat['course_id'] = train['course_id'].str.split(' ')
course_mat = list_value_to_column(course_mat, col='course_id')
print('query_mat :', query_mat.shape)
print('course_mat :', course_mat.shape)
sum = np.array(query_mat.sum())
sum = np.tile(sum, (664, 1))
appearence = np.dot(course_mat.T.to_numpy(), query_mat.to_numpy())

if args.smooth_method=='add_one':

    sum = sum+1
    appearence = appearence+1
    smooth = appearence/sum
else:
    prob = appearence/sum
    lamda = 0.5
    smooth = prob*lamda+(1-lamda)*(sum/59737)
valid = list_value_to_column(valid)
valid2 = pd.DataFrame()
empty = 0
for i in query_mat.columns:
    try:
        valid2[i] = valid[i]
    except:
        # print(i)
        empty+=1
        valid2[i] = 1
valid_mat = valid2.to_numpy()
valid_course_prob = np.dot(valid_mat, np.log(np.transpose(smooth)))
def get_topk_QLM(course_score, course_list, k):
    score = course_score
    list1, list2 = zip(*sorted(zip(score, course_list), reverse=True))
    return list2[:k]

predict = [get_topk_QLM(i, courses['course_id'], 50) for i in valid_course_prob]
actual = list(valid['course_id'].str.split())
print(mapk(actual, predict, 50))
