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

def plot_performance(x, y1, y2, y3, matrix=''):
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.title(matrix + ' curves (All)')
    plt.xlabel('steps') #set the label for x-axis
    plt.ylabel(matrix) #set the label for y axis
    plt.legend(['roberta wwm ext large', 'bert', 'unpretrain roberta wwm ext large'])
    os.makedirs('image', exist_ok=True)
    plt.savefig(f"image/{matrix}_all.png")
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


    path_bert_1 = os.path.join("/", "home", 'maomao', "Downloads", "trainer_state_bb.json")
    performance_bert_1 = load_json(path_bert_1)
    performance_df_bert_1 = pd.DataFrame.from_dict(performance_bert_1)

    log_history_bert_1 = performance_df_bert_1['log_history'].tolist()
    loss_bert_1 = pd.DataFrame(log_history_bert_1)['loss'].dropna().tolist()
    n_record_bert_1 = len(loss_bert_1)


    path_bert_2 = os.path.join("/", "home", 'maomao', "Downloads", "trainer_state_bert.json")
    performance_bert_2 = load_json(path_bert_2)
    performance_df_bert_2 = pd.DataFrame.from_dict(performance_bert_2)

    log_history_bert_2 = performance_df_bert_2['log_history'].tolist()
    EM_bert_2 = pd.DataFrame(log_history_bert_2)['eval_exact_match'].dropna().tolist()
    EM_bert_2 = [elem / 100.0 * 10 for elem in EM_bert_2]
    loss_bert_2 = pd.DataFrame(log_history_bert_2)['loss'].dropna().tolist()
    n_record_bert_2 = min(len(EM_bert_2), len(loss_bert_2))

    path_unpretrain = os.path.join("/", "home", 'maomao', "Downloads", "unpretrain_trainer_state.json")
    performance_unpretrain = load_json(path_unpretrain)
    performance_df_unpretrain = pd.DataFrame.from_dict(performance_unpretrain)

    log_history_unpretrain = performance_df_unpretrain['log_history'].tolist()
    EM_unpretrain = pd.DataFrame(log_history_unpretrain)['eval_exact_match'].dropna().tolist()
    EM_unpretrain = [elem / 100.0 for elem in EM_unpretrain]
    loss_unpretrain = pd.DataFrame(log_history_unpretrain)['loss'].dropna().tolist()
    n_record_unpretrain = min(len(EM_unpretrain), len(loss_unpretrain))

    n_record = min(n_record, n_record_bert_1, n_record_bert_2, n_record_unpretrain)

    steps = [500 * idx for idx in range(n_record)]

    plot_performance(steps[:n_record], loss[:n_record], loss_bert_1[:n_record], loss_unpretrain[:n_record], matrix='loss')
    plot_performance(steps[:n_record], EM[:n_record], EM_bert_2[:n_record], EM_unpretrain[:n_record], matrix='EM')

if __name__ == "__main__":
    main()

