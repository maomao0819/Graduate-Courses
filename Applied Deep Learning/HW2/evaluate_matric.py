import os
import pandas as pd
import json
import matplotlib.pyplot as plt

def load_json(json_path):
    if (json_path is not None) and os.path.exists(json_path):
        print(f'[*] Loading {json_path}...', end='', flush=True)
        with open(json_path, 'r') as f:
            result = json.load(f)
        print('done')

        return result

def plot_performance(x, y, matrix=''):
    plt.plot(x, y)
    plt.title(matrix + ' curves')
    plt.xlabel('steps') #set the label for x-axis
    plt.ylabel(matrix) #set the label for y axis
    os.makedirs('image', exist_ok=True)
    plt.savefig(f"image/{matrix}.png")
    plt.clf()

def main():
    path = os.path.join('model', 'hfl-chinese-roberta-wwm-ext-large', 'question-answering', 'trainer_state.json')
    performance = load_json(path)
    performance_df = pd.DataFrame.from_dict(performance)

    log_history = performance_df['log_history'].tolist()
    EM = pd.DataFrame(log_history)['eval_exact_match'].dropna().tolist()
    EM = [elem / 100.0 for elem in EM]
    loss = pd.DataFrame(log_history)['loss'].dropna().tolist()
    n_record = min(len(EM), len(loss))
    steps = [500 * idx for idx in range(n_record)]

    plot_performance(steps[:n_record], loss[:n_record], matrix='loss')
    plot_performance(steps[:n_record], EM[:n_record], matrix='EM')

if __name__ == "__main__":
    main()

