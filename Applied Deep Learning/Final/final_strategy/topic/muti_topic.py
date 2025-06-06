import torch
import logging
from transformers import AutoTokenizer
from argparse import ArgumentParser, Namespace
import pandas as pd
import numpy as np
from torch.nn import Embedding
from transformers import BertModel,AutoConfig
from tqdm import trange,tqdm
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers.trainer_utils import get_last_checkpoint
from transformers import EvalPrediction
import os
from datasets import Dataset
import sys
sys.path.append('..')
from utils import *

def main(args):
    output_dir = args.output_dir
    logger = logging.getLogger(__name__)

    group = load_from_pickle(os.path.join(args.data_dir,"subgroup"))
    user = load_from_pickle(os.path.join(args.data_dir,"user"))
    group_list = group.subgroup_name.to_list()
    id2label = {i:k for i,k in enumerate(group_list,1)}
    label2id = {k:i for i,k in enumerate(group_list,1)}
    # course = pd.read_csv("./data/courses.csv",usecols = ["course_id","course_name"])
    # course_list = course.course_name.to_list() 
    # id2label = {i:k for i,k in enumerate(course_list)}
    # label2id = {k:i for i,k in enumerate(course_list)}


    device = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    df_train = pd.read_csv(args.train_file)
    df_eval = pd.read_csv(args.validation_file)
    df_train = convert_train(df_train,user)
    df_eval = convert_train(df_eval,user)
    
    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)
    column_name = [f"group_{i}" for i in range(1,92)]
    # column_name = [f"course_{i}" for i in range(728)]

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

    def preprocess_data(examples):
        text = examples["text"]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)

        labels_batch = {k: examples[k] for k in examples.keys() if k in column_name}
        labels_matrix = np.zeros((len(text), len(column_name)))
        for idx, label in enumerate(column_name):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()

        return encoding

    last_checkpoint = None
    if os.path.isdir(output_dir) :
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    train_encoded = train_dataset.map(preprocess_data, batched=True,remove_columns = train_dataset.column_names)
    train_encoded.set_format("torch")

    eval_encoded = eval_dataset.map(preprocess_data, batched=True,remove_columns = eval_dataset.column_names)
    eval_encoded.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", 
                                                            problem_type="multi_label_classification", 
                                                            num_labels=len(column_name),
                                                            id2label=id2label,
                                                            label2id=label2id)
    batch_size = 12
    metric_name = "f1"



    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=25,
        gradient_accumulation_steps = 3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        #push_to_hub=True,
        seed = 26
    )

    def multi_label_metrics(predictions, labels, threshold=0.15):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        y_true_ids = [[ i for i,k in enumerate(sublist) if k == 1.0 ] for sublist in y_true]
        y_pred_ids = [[ i for i,k in enumerate(sublist) if k == 1.0 ] for sublist in y_pred]
    
        
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy,
                'ACCCC:':mapk(y_true_ids,[ i[::-1] for i in np.argsort(probs).tolist()],50)
                }
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)

        return result

    trainer = Trainer(
        model,
        args,
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        tokenizer=tokenizer,            
        compute_metrics=compute_metrics
    )

    checkpoint = None

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.evaluate()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output_dir", default='./tmp', type=str, help="the path to sava checkpoint.")
    parser.add_argument("--data_dir", default='../data', type=str, help="the directory to data.")
    parser.add_argument("--train_file", default=None, type=str, help="the path to train file.")
    parser.add_argument("--validation_file", default=None, type=str, help="the path to validation_file.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)