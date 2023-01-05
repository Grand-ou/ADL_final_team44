import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np


class Tool():
    def __init__(
        self,
        courses_file,
        train_file,
        test_seen_file,
        test_seen_topic_file,
        subgroups_file,
        pred_topic,
    ):
        df_courses = pd.read_csv(courses_file)
        df_train = pd.read_csv(train_file)
        df_test_seen = pd.read_csv(test_seen_file)

        self.course2idx = {
            id: idx
            for (idx, id) in enumerate(df_courses['course_id'])
        }
        self.idx2course = {
            idx: id
            for (idx, id) in enumerate(df_courses['course_id'])
        }

        self.user2idx = {
            id: idx
            for (idx, id) in enumerate(df_train['user_id'])
        }
        self.idx2user = {
            idx: id
            for (idx, id) in enumerate(df_train['user_id'])
        }

        num_items = len(self.course2idx)
        num_users = len(self.user2idx)
        num_records = 0
        purch_hists = []
        row_ind = []
        col_ind = []
        for i in range(num_users):
            purch_hist = df_train.iloc[i]['course_id'].split(' ')
            purch_hist = [self.course2idx[j] for j in purch_hist]
            purch_hists.append(purch_hist)
            num_purch = len(purch_hist)
            row_ind += [i]*num_purch
            col_ind += purch_hist
            num_records += num_purch
        
        self.purch_hists = purch_hists
        self.user_item_data = csr_matrix(
            (np.ones(num_records),
            (row_ind, col_ind)),
            shape=(num_users, num_items)
        )
        self.user_ids_test = [self.user2idx[i] for i in df_test_seen['user_id']]

        if pred_topic:
            df_subgroup = pd.read_csv(subgroups_file)
            df_test_seen_group = pd.read_csv(test_seen_topic_file)
            subgroup2idx = {
                name: idx
                for (name, idx) in zip(df_subgroup['subgroup_name'], df_subgroup['subgroup_id'])
            }

            course2subgroup = {}
            for (idx, subgroup) in enumerate(df_courses['sub_groups']):
                try:
                    course2subgroup[idx] = [subgroup2idx[i] for i in subgroup.split(',')]
                except:
                    course2subgroup[idx] = [0]
            self.course2subgroup = course2subgroup