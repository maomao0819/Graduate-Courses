import pandas as pd
import pickle
import numpy as np

def write_to_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def convert_train(df,user):
    df.fillna('',inplace=True)
    info = ["interests","recreation_names"]
    ids = [id for id in df['user_id']]
    interests = ["".join(user.loc[id][info]).replace(',',' ') for id in ids] 
    text = [interests[i] for i,_ in enumerate(ids)]
    df["text"] = text

    column_name = [f"group_{i}" for i in range(1,92)]
    label = df["subgroup"].to_list()
    labels = [np.fromstring(label_id,dtype=int,sep = ' ') for label_id in label]
    l = [[1 if i in sublist else 0 for i in range(1,92)] for sublist in labels]
    df[column_name] = l
    return df
    
def convert_test(df,user):
    df.fillna('',inplace=True)
    info = ["interests","recreation_names"]
    ids = [id for id in df['user_id']]
    interests = ["".join(user.loc[id][info]).replace(',',' ') for id in ids] 
    text = [interests[i] for i,_ in enumerate(ids)]
    df["text"] = text
    return df

def apk(actual, predicted, k=10):
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
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])