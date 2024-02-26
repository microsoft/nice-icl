import os
# import sys
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
# sys.path.append(os.path.abspath(os.path.join('..')))

seed_value = 42
np.random.seed(seed_value)
class SST2:

    def __init__(self, name, config=None):
        self.name = name
        # self.data_path = os.path.abspath(os.path.join('..')) + "/data/classification/sst5/"
        self.train_data = self.load_data('./train.tsv')
        self.test_data =  self.load_data('./dev.tsv')
        self.test_data = self.test_data if len(self.test_data)<=1000 else np.random.choice(self.test_data, size=1000, replace=False)
        
        self.config = config
        
    def load_data(self, path):
        # read tsv files and return a list of dicts
        data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            # skip the first line
            lines = lines[1:]
            for line in lines:
                line = line.split('\t')
                data.append({'text': line[0], 'label': int(line[1])})
        return data


    def data(self):

        return self.train_data, self.test_data