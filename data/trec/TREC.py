from datasets import load_dataset
import numpy as np


class TREC:

    def __init__(self, name, config):
        self.name = name
        dataset_train = load_dataset("trec", split="train")
        dataset_train = dataset_train if len(dataset_train)<=10000 else np.random.choice(dataset_train, size=10000, replace=False)
        dataset_test = load_dataset("trec", split="test")
        dataset_test = dataset_test if len(dataset_test)<=1000 else np.random.choice(dataset_test, size=1000, replace=False)
        
        num2label = {"0": "Abbreviation", "1": "Entity", "2": "Description", "3": "Human", "4": "Location", "5": "Number"}

        train_labels = []
        for i in range(len(dataset_train)):
            train_labels.append(num2label[str(dataset_train[i]['coarse_label'])])
        self.train_data = [{'text': dataset_train[i]['text'], 'label': train_labels[i]} for i in range(len(train_labels))]

        test_labels = []
        for i in range(len(dataset_test)):
            test_labels.append(num2label[str(dataset_test[i]['coarse_label'])])
        self.test_data = [{'text': dataset_test[i]['text'], 'label': test_labels[i]} for i in range(len(test_labels))]

        self.config = config


    def data(self):

        return self.train_data, self.test_data
