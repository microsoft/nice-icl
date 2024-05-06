from datasets import load_metric
import  argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='text', help='path to the file to be evaluated')
parser.add_argument('--show', action='store_false')

args = parser.parse_args()

def evaluate(preds, golds):
    # character-level 4-bleu
    bleu = load_metric('bleu')
    predictions = [[ch for ch in text] for text in preds]
    references = [[[ch for ch in entry]] for entry in golds]
    return bleu.compute(predictions=predictions, references=references)


df = pd.read_csv(args.path)

scores = []
for index, row in df.iterrows():
    
    pred = [row['prediction'].strip()]
    ref = [row['ground_truth'].strip()]
    
    score = evaluate(pred, ref)    
    scores.append(score['bleu'])
    
df['correct'] = scores

acc = df['correct'].mean()

if  args.show:
    print(f"Accuracy:", acc)

df.to_csv(args.path)
    