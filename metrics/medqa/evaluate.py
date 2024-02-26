import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='text', help='path to the file to be evaluated')
parser.add_argument('--show', action='store_false')

args = parser.parse_args()

df = pd.read_csv(args.path)

df['ground_truth'] = df['ground_truth'].apply(lambda x: str(x).strip().lower())
df['prediction'] = df['prediction'].apply(lambda x: str(x).split(":")[1].strip().lower() if str(x).find(":") >=0 else str(x).strip().lower())

df['correct'] = (df['ground_truth'] == df['prediction']).astype(int)

acc = df['correct'].mean()

if  args.show:
    print(f"Accuracy:", acc)
