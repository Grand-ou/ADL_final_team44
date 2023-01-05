import os
import csv

import pandas as pd
import time
import argparse
from rank_bm25 import BM25Okapi
from ckiptagger import WS
parser = argparse.ArgumentParser()
parser.add_argument("data_path", default='add_one', help="data folder path")
parser.add_argument("ckip_path", default='./data', help="data of ckiptagger folder path")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ws = WS("./data", disable_cuda=False)
# data_utils.download_data_gdown("./")

with open('courses_doc.csv', newline='') as f:
    reader = csv.reader(f)
    courses_document = list(reader)
data_path = 'hahow/data/'
courses = pd.read_csv(data_path+'courses.csv')
users = pd.read_csv(data_path+'users.csv')
# train = pd.read_csv(data_path+'train.csv')
# valid = pd.read_csv(data_path+'val_unseen.csv')
test = pd.read_csv(data_path+'test_unseen.csv')
users['gender'] = users['gender'].map({'male': '男', 'female': '女', 'other': '三性'})
# valid = valid.merge(users, how='left', on='user_id')
test = test.merge(users, how='left', on='user_id')
# train = train.merge(users, how='left', on='user_id')
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
test = query_genarate(test)
bm25Model = BM25Okapi(courses_document)
def get_topk_prediction(bmmodel, query_list, course_list, k):
    score = bmmodel.get_scores(query_list)
    list1, list2 = zip(*sorted(zip(score, course_list), reverse=True))
    return list2[:k]


predict = [get_topk_prediction(bm25Model, i, courses['course_id'], 50) for i in test['query_list'].tolist()]
with open('courses_submission_unseen.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    sub = [[test['user_id'][i], ' '.join(list(predict[i]))] for i in range(len(predict))]
    writer.writerow(['user_id', 'course_id'])
    writer.writerows(sub)