import json
from typing import Dict
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer
def save_jsonl(ids, predictions, json_path):
    if json_path.count("/"):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='UTF-8') as fp:
        prediction_summery = {}
        n_records = len(ids)
        for record_id in range(len(ids)):
            prediction_summery['title'] = predictions[record_id]
            prediction_summery['id'] = ids[record_id]
            json.dump(prediction_summery, fp, ensure_ascii=False)
            fp.write('\n')

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
        for batch_idx, data in enumerate(tqdm_loop, 1):
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
        