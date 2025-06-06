import numpy as np
import pandas as pd
import csv
from argparse import ArgumentParser, Namespace
from collections import Counter
import sys
sys.path.append("..")
import random
from utils import *


def main(args):
    # train_data = pd.read_csv("./hahow/data/train.csv")
    # val_seen = pd.read_csv("./hahow/data/val_seen.csv")
    # test_seen = pd.read_csv("./hahow/data/test_seen.csv")
    # courses = pd.read_csv("./hahow/data/courses.csv")
    # users = pd.read_csv("./hahow/data/users.csv")

    train_data = load_from_pickle('../data/train')
    val_seen = load_from_pickle('../data/val_seen')
    test_seen = pd.read_csv(args.test_file)
    courses = load_from_pickle("../data/course")
    users = load_from_pickle('../data/user')



    def get_course_list(course_data):
        course_list = list()
        for id, row in course_data.iterrows():
            course_list.append(id)
        return course_list

    def count_purchase(data):
        d = dict()
        for _, row in data.iterrows():
            course_ids = row['course_id'].split(" ")
            for course_id in course_ids:
                if course_id not in d:
                    d[course_id] = 1
                else:
                    d[course_id] += 1
        counter = Counter(d)
        return counter.most_common(50)

    def get_users_list(users_data):
        users_list = list()
        for _, row in users_data.iterrows():
            users_list.append(row["user_id"])
        return users_list

    course_list = get_course_list(courses)
    c = count_purchase(val_seen)
    #c = count_purchase(train_data)

    candidate_list = [pair[0] for pair in c]
    #candidate_list = course_list[650:] + candidate_list
    #print(len(candidate_list))
    #print(candidate_list)
    top_list = list()
    upper = 0
    lower = 0
    for candidate in candidate_list[:50]:
        top_list.append(course_list.index(candidate))
        if course_list.index(candidate) > 600:
            upper += 1
        else:
            lower +=1
    #print(top_list[:50])
    #print(upper, lower)
    test_users_list = get_users_list(test_seen)
    res = list()
    for test_user in test_users_list:
        tmp = list()
        tmp.append(test_user)
        #random.shuffle(candidate_list)
        res.append([test_user, " ".join(candidate_list[:50])])
    #print(res)
    with open("./seen_course.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "course_id"])
        for data in res:
            writer.writerow(data)

    print("Finish writing to seen_course.csv")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", default=None, type=str, help="the path to test file.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)