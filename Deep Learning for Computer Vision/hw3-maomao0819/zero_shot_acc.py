import parser
import numpy as np
import pandas as pd

def accuracy(args):
    df = pd.read_csv(args.predict_path)
    prediction = df['label'].tolist()
    real_label = df['filename'].apply(lambda x: int(x.split('_')[0])).tolist()
    acc = np.sum(np.array(prediction) == np.array(real_label)) / len(prediction)
    print(f"{100 * acc:.2f}%")

if __name__ == "__main__":
    args = parser.arg_parse(1)
    accuracy(args)