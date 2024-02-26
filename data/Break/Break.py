from datasets import load_dataset
import numpy as np

class Break:
    def __init__(self, name, config=None):
        
        self.name = name
        dataset_train = load_dataset("break_data", "QDMR", split="train")
        dataset_train = dataset_train if len(dataset_train)<=10000 else np.random.choice(dataset_train, size=10000, replace=False)
        dataset_test = load_dataset("break_data", "QDMR", split="validation")
        dataset_test = dataset_test if len(dataset_test)<=1000 else np.random.choice(dataset_test, size=1000, replace=False)
        
        
        train_text = []
        train_labels = []
        for i in range(len(dataset_train)):
            decomp = dataset_train[i]['decomposition']
            train_question = dataset_train[i]["question_text"]
            train_text.append(train_question)
            train_labels.append(decomp)

        self.train_data = [{'text': train_text[i], 'label': train_labels[i]} for i in range(len(train_text))]

        test_text = []
        test_labels = []
        for i in range(len(dataset_test)):
            decomp = dataset_test[i]['decomposition']
            test_question = dataset_test[i]["question_text"]
            test_text.append(test_question)
            test_labels.append(decomp)


        self.test_data = [{'text': test_text[i], 'label': test_labels[i]} for i in range(len(test_text))]

        self.config = config
        
    def data(self):
        return self.train_data, self.test_data
