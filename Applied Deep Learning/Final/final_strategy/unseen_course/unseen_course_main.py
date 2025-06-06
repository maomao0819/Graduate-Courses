import pandas as pd
import jsonlines
import numpy as np
import csv
from tqdm import tqdm, trange
from argparse import ArgumentParser,Namespace
import sys
sys.path.append("..")
from utils import *
from sentence_transformers import SentenceTransformer,util

def main(args):
#Read file
    tokenizer = SentenceTransformer('shibing624/text2vec-base-chinese')
    user = load_from_pickle('../data/user')
    course = load_from_pickle('../data/course')
    val_unseen = load_from_pickle('../data/val_unseen')
    es = [x.split(' ') for x in val_unseen['course_id'].values]
    id2index = {k:i for i,k in enumerate(course.index.values)}
    val_unseen_id = [list(map(lambda x:id2index[x], x)) if x != [''] else [] for x in es]
    val_unseen_id_class =  [item for sublist in val_unseen_id for item in sublist]
    eval_unseen_class_stat = [0]*728
    for i in val_unseen_id_class:
        eval_unseen_class_stat[i]+=1
    eval_unseen_top = np.argsort(eval_unseen_class_stat)[::-1][:7]
    df_test = pd.read_csv(args.test_file)
    df_test = convert_test(df_test,user)
    de = df_test['text'].to_list()
    eval_embeddings = [tokenizer.encode(i) for i in tqdm(de)]
    course_list = course.course_name.to_list()

# set the candidate course by course_filter result
# get target course embeddings
    candidate = list(set(eval_unseen_top))
    course_top = [course_list[i] for i in candidate]
    print(course_top)
    course_embeddings = [tokenizer.encode(i) for i in tqdm(course_top)]
#predict
    ans = []
    for i,k in tqdm(enumerate(eval_embeddings)):
        prob_index = util.pytorch_cos_sim(eval_embeddings[i],course_embeddings).squeeze(0).argsort(descending=True).tolist()
        prob = [candidate[i] for i in prob_index]
        ans.append(list(dict.fromkeys(prob))[:])
    ans_ = [list(map(lambda x:list(course.index)[x],ans[i])) for i in range(len(ans))]
# delete seen course from predict
    seen = load_from_pickle('../data/train')
    seen = {i:k.split(' ') for i,k in zip(seen.user_id,seen.course_id)}
    for i,user in enumerate(df_test['user_id'].values):
        if(user in seen.keys()):
            ans_[i] = [x for x in ans_[i] if x not in seen[user]]
    ans_ = [" ".join(item) for item in ans_]
#Write to csv
    data = {
    'user_id':df_test['user_id'].to_list(),
    'course_id':ans_
    }
    pd.DataFrame(data).to_csv('./unseen_course.csv',index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", default=None, type=str, help="the path to test file.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)