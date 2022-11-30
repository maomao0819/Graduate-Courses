import json
from typing import Dict
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer
# from rouge_score import rouge_scorer, scoring

def save_jsonl(ids, predictions, json_path):
    if json_path.count("/"):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='UTF-8') as fp:
        prediction_summery = {}
        for record_id in range(len(ids)):
            prediction_summery['title'] = predictions[record_id]
            prediction_summery['id'] = ids[record_id]
            json.dump(prediction_summery, fp, ensure_ascii=False)
            fp.write('\n')

def load_json(json_path):
    if (json_path is not None) and os.path.exists(json_path):
        print(f'[*] Loading {json_path}...', end='', flush=True)
        with open(json_path, 'r') as f:
            result = json.load(f)
        print('done')

        return result

def generate(
    model: Dict[str, torch.nn.Module],
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    do_sample: bool,
    num_beams: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_length: int,
):
    model.eval()
    generations = []
    n_batch = len(dataloader)
    tqdm_loop = tqdm((dataloader), total=n_batch)
    with torch.no_grad():
        for data in tqdm_loop:
            # ref: https://huggingface.co/blog/how-to-generate
            generation = model.generate(
                data["input_ids"].to(device),
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length,
            )
            generations.extend(generation)
        generations = tokenizer.batch_decode(
                    generations, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
    return generations

# def compute_score(predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False, tokenizer=None):
#     if rouge_types is None:
#         rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

#     multi_ref = isinstance(references[0], list)

#     if tokenizer is not None:
#         tokenizer = Tokenizer(tokenizer)

#     scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
#     if use_aggregator:
#         aggregator = scoring.BootstrapAggregator()
#     else:
#         scores = []

#     for ref, pred in zip(references, predictions):
#         if multi_ref:
#             score = scorer.score_multi(ref, pred)
#         else:
#             score = scorer.score(ref, pred)
#         if use_aggregator:
#             aggregator.add_scores(score)
#         else:
#             scores.append(score)

#     if use_aggregator:
#         result = aggregator.aggregate()
#         for key in result:
#             result[key] = result[key].mid.fmeasure

#     else:
#         result = {}
#         for key in scores[0]:
#             result[key] = list(score[key].fmeasure for score in scores)

#     return result