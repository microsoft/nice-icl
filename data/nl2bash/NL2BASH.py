from datasets import load_dataset
import numpy as np

class NL2BASH:

    def __init__(self, name, config):
        self.name = name
        dataset_train = load_dataset("jiacheng-ye/nl2bash", split='train')
        dataset_train = dataset_train if len(dataset_train)<=10000 else np.random.choice(dataset_train, size=10000, replace=False)
        dataset_test = load_dataset("jiacheng-ye/nl2bash", split='validation')
        dataset_test = dataset_test if len(dataset_test)<=1000 else np.random.choice(dataset_test, size=1000, replace=False)

        self.train_nl = []
        self.train_bash = []

        for i in range(len(dataset_train)):
            self.train_nl.append(dataset_train[i]['nl'])
            self.train_bash.append(dataset_train[i]['bash'])
        self.train_data = [{'text': self.train_nl[i], 'label': self.train_bash[i]} for i in range(len(self.train_nl))]

        self.test_nl = []
        self.test_bash = []

        for i in range(len(dataset_test)):
            self.test_nl.append(dataset_test[i]['nl'])
            self.test_bash.append(dataset_test[i]['bash'])
        self.test_data = [{'text': self.test_nl[i], 'label': self.test_bash[i]} for i in range(len(self.test_nl))]

        self.config = config


    def data(self):

        return self.train_data, self.test_data
