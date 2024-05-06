import pandas as pd
from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
import re
import glob
from functools import partial
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='text', help='path to the file to be evaluated')
parser.add_argument('--show', action='store_false')

args = parser.parse_args()

logger = logging.getLogger(__name__)

def wrapper(idx_args, func):
    idx, args = idx_args
    res = func(args)
    return idx, res

def parallel_run(func, args_list, n_processes=8, initializer=None, initargs=None, **kwargs):
    idx2res = {}
    func = partial(func, **kwargs)
    n = len(args_list)
    logger.info(f"Parallel running with {n_processes} processes")
    with Pool(n_processes, initializer=initializer, initargs=initargs) as p:
        for idx, response in tqdm(p.imap_unordered(partial(wrapper, func=func),
                                                   enumerate(args_list)),
                                  total=n):
            idx2res[idx] = response

    res = [idx2res[i] for i in range(n)]
    return res

def eval_single(args):
    pred, gold = args
    return str(pred).strip() == gold


df = pd.read_csv(args.path)
df['output'] = df['output'].apply(lambda x: str(x).lower())
df['answers'] = df['answers'].apply(lambda x: x.lower())
print(df['output'])
preds = df['output'].tolist()
golds = df['answers'].tolist()

eval_results = parallel_run(eval_single, list(zip(preds, golds)))

df['EM']  = eval_results
df.to_csv(args.path)

acc = df['EM'].mean()

if  args.show:
    print(f"EM Score:", acc)
