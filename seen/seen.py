import argparse
import pandas as pd
import numpy as np
from implicit.gpu.als import AlternatingLeastSquares

from utils import (
    mapk,
    predict,
    predict_topic_from_course
)
from tools import Tool


def main(args):
    tool = Tool(
        args.courses_file,
        args.train_file,
        args.test_seen_file,
        args.test_seen_topic_file,
        args.subgroups_file,
        args.pred_topic
    )
    
    model_als = AlternatingLeastSquares(
        factors=1000,
        regularization=200,
        # alpha=1000,
        iterations=1,
        random_state=11112224
    )
    model_als.fit(tool.user_item_data)


    if args.pred_topic:
        predg_als_test, scoreg_als_test = model_als.recommend(
            tool.user_ids_test,
            tool.user_item_data[tool.user_ids_test],
            filter_already_liked_items=False,
            N=50
        )
        predg_als_test = predict_topic_from_course(
            result=predg_als_test,
            user_ids=tool.user_ids_test,
            purch_hists=tool.purch_hists,
            course2subgroup=tool.course2subgroup
        )
        predict(
            result=predg_als_test, 
            path=args.topic_pred_file,
            user_ids=tool.user_ids_test,
            idx2user=tool.idx2user,
            domain='topic'
        )
    else:
        pred_als_test, score_als_test = model_als.recommend(
            tool.user_ids_test,
            tool.user_item_data[tool.user_ids_test],
            N=50
        )
        pred_als_test = [[tool.idx2course[j] for j in i] for i in pred_als_test]
        predict(
            result=pred_als_test, 
            path=args.course_pred_file,
            user_ids=tool.user_ids_test,
            idx2user=tool.idx2user
        )
        
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        default='data/train.csv'
    )
    parser.add_argument(
        "--courses_file",
        type=str,
        default='data/courses.csv'
    )
    parser.add_argument(
        "--subgroups_file",
        type=str,
        default='data/subgroups.csv'
    )
    parser.add_argument(
        "--test_seen_file",
        type=str,
        default='data/test_seen.csv'
    )
    parser.add_argument(
        "--test_seen_topic_file",
        type=str,
        default='data/test_seen_group.csv'
    )
    parser.add_argument(
        "--course_pred_file",
        type=str,
        default='seen_course_pred.csv'
    )
    parser.add_argument(
        "--topic_pred_file",
        type=str,
        default='seen_topic_pred.csv'
    )
    parser.add_argument(
        "--pred_course",
        action="store_true"
    )
    parser.add_argument(
        "--pred_topic",
        action="store_true"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)