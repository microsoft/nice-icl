import numpy as np
from datasets import load_dataset


seed_value = 42
np.random.seed(seed_value)
class SST5:

    def __init__(self, name, config=None):
        self.name = name
        # self.data_path = os.path.abspath(os.path.join('..')) + "/data/classification/sst5/"
        self.dataset = load_dataset("SetFit/sst5")
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["validation"]
        self.train_data = self.train_data if len(self.train_data)<=1000 else np.random.choice(self.train_data, size=1000, replace=False)
        self.test_data = self.test_data if len(self.test_data)<=1000 else np.random.choice(self.test_data, size=1000, replace=False)
        self.config = config


    def data(self):

        return self.train_data, self.test_data