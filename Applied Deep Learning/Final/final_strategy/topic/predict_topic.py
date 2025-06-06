import torch
import pandas as pd
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from torch.nn import Embedding
from transformers import BertModel,AutoConfig 
from transformers import AutoModelForSequenceClassification
from tqdm import trange,tqdm
from argparse import ArgumentParser, Namespace
import csv
import os
from datasets import Dataset
import sys;sys.path.append('..')
from utils import *

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    user = load_from_pickle(os.path.join(args.data_dir,"user"))
    row_test = pd.read_csv(args.test_file)

    group = load_from_pickle(os.path.join(args.data_dir,"subgroup"))
    group_list = group.subgroup_name.to_list()
    id2label = {i:k for i,k in enumerate(group_list,1)}
    label2id = {k:i for i,k in enumerate(group_list,1)}

    device = 'cuda'
    row_test = convert_test(row_test,user)
    test_dataset = Dataset.from_pandas(row_test)
    column_name = [f"group_{i}" for i in range(1,92)]
    def preprocess_data(examples):
        text = examples["text"]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=256)
        return encoding
    test_encoded = test_dataset.map(preprocess_data, batched=True,remove_columns = test_dataset.column_names)
    test_encoded.set_format("torch")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(column_name),
                                                           id2label=id2label,
                                                           label2id=label2id)
    model = model.to("cuda")
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_encoded, batch_size=24)
    ans = []
    for data in tqdm(test_loader):
        encoding = {k: v.to("cuda") for k,v in data.items()}
        outputs = model(**encoding)
        logits = outputs.logits
        probs  = torch.sigmoid(logits).cpu().detach().numpy()
        ans.extend([i[::-1] for i in (np.argsort(probs)+1).tolist()]) 
    ans = [list(i)[:] for i in ans]
    ans_ = [' '.join([str(j) for j in i]) for i in ans]
    data = {
        'user_id':row_test['user_id'].to_list(),
        'subgroup':ans_
    }
    pd.DataFrame(data).to_csv(args.output_dir,index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output_dir", default='./predict.csv', type=str, help="the path to output.")
    parser.add_argument("--data_dir", default='./data', type=str, help="the directory to data.")
    parser.add_argument("--test_file", default=None, type=str, help="the path to test file.")
    parser.add_argument("--model_name_or_path", default='./data/checkpoint-25000', type=str, help="the path to model.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)