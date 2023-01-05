import os
import csv
import argparse
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MultiLabelBinarizer

from ckiptagger import WS
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
# valid = pd.read_csv(data_path+'val_unseen.csv')
test = pd.read_csv(data_path+'test_unseen.csv')
users['gender'] = users['gender'].map({'male': '男', 'female': '女', 'other': '三性'})
# valid = valid.merge(users, how='left', on='user_id')
test = test.merge(users, how='left', on='user_id')
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
# valid = query_genarate(valid)
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
test = list_value_to_column(test)
test2 = pd.DataFrame()
empty = 0
for i in query_mat.columns:
    try:
        test2[i] = test[i]
    except:
        # print(i)
        empty+=1
        test2[i] = 1
test_mat = test2.to_numpy()
test_course_prob = np.dot(test_mat, np.log(np.transpose(smooth)))
def get_topk_QLM(course_score, course_list, k):
    score = course_score
    list1, list2 = zip(*sorted(zip(score, course_list), reverse=True))
    return list2[:k]

predict = [get_topk_QLM(i, courses['course_id'], 100) for i in test_course_prob]

with open('courses_submission_unseen.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    sub = [[test['user_id'][i], ' '.join(list(predict[i]))] for i in range(len(predict))]
    writer.writerow(['user_id', 'course_id'])
    writer.writerows(sub)