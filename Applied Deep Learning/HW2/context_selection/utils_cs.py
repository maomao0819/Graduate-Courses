import os
import json

def load_json(json_path):
    if (json_path is not None) and os.path.exists(json_path):
        print(f'[*] Loading {json_path}...', end='', flush=True)
        with open(json_path, 'r') as f:
            result = json.load(f)
        print('done')

        return result

def save_json(data, json_path):
    print(f'[*] Saving to {json_path}...', end='', flush=True)
    if not json_path.endswith(".json"):
        json_path = os.path.join(json_path, 'predict_cs.json')
    if json_path.count("/"):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # with open(json_path, 'w') as f:
    #     json.dump(data, f)
    json.dump(data, open(json_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('done')