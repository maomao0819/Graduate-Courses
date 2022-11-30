import os
import pandas as pd
import utils
import matplotlib.pyplot as plt

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
    
    path = os.path.join('summarization_weight', 'trainer_state.json')
    performance = utils.load_json(path)
    performance_df = pd.DataFrame.from_dict(performance)

    log_history = performance_df['log_history'].tolist()
    loss = pd.DataFrame(log_history)['loss'].dropna().tolist()
    eval_loss = pd.DataFrame(log_history)['eval_loss'].dropna().tolist()
    steps = [500 * idx for idx in range(len(loss))]

    os.makedirs('image', exist_ok=True)

    plt.plot(steps, loss)
    plt.plot(steps, eval_loss)
    plt.title("loss")
    plt.xlabel('steps') #set the label for x-axis
    plt.ylabel("loss") #set the label for y axis
    plt.legend(['training loss', 'evaluation loss'])
    plt.savefig(f"image/loss.png")
    plt.clf()

    eval_rouge1 = pd.DataFrame(log_history)['eval_rouge1'].dropna().tolist()
    eval_rouge2 = pd.DataFrame(log_history)['eval_rouge2'].dropna().tolist()
    eval_rougeL = pd.DataFrame(log_history)['eval_rougeL'].dropna().tolist()

    plt.plot(steps, eval_rouge1)
    plt.plot(steps, eval_rouge2)
    plt.plot(steps, eval_rougeL)
    plt.title("rouge")
    plt.xlabel('steps') #set the label for x-axis
    plt.ylabel("rouge") #set the label for y axis
    plt.legend(['rouge1', 'rouge2', 'rougeL'])
    plt.savefig(f"image/rouge.png")
    plt.clf()

if __name__ == "__main__":
    main()

