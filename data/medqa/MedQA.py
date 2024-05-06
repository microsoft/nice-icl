from datasets import load_dataset
import numpy as np

class MedQA:

    def __init__(self, name, config=None):

        self.name = name
        dataset_train = load_dataset("bigbio/med_qa", 'med_qa_en_4options_bigbio_qa', split="train", trust_remote_code=True)
        dataset_train = dataset_train if len(dataset_train)<=10000 else np.random.choice(dataset_train, size=10000, replace=False)
        dataset_test = load_dataset("bigbio/med_qa", 'med_qa_en_4options_bigbio_qa', split="validation", trust_remote_code=True)
        dataset_test = dataset_test if len(dataset_test)<=1000 else np.random.choice(dataset_test, size=1000, replace=False)

        
        train_text = []
        train_labels = []
        for i in range(len(dataset_train)):
            choices = dataset_train[i]['choices']
            train_question = dataset_train[i]["question"]
            choices = f"A: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}"
            train_labels.append(dataset_train[i]['answer'][0])
            text = f"question: {train_question}\nchoices:\n{choices}"
            train_text.append(text)

        self.train_data = [{'text': train_text[i], 'label': train_labels[i]} for i in range(len(train_text))]

        test_text = []
        test_labels = []
        for i in range(len(dataset_test)):
            choices = dataset_test[i]['choices']
            test_question = dataset_test[i]["question"]
            choices = f"A: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}"
            test_labels.append(dataset_test[i]['answer'][0])
            text = f"question: {test_question}\nchoices:\n{choices}"
            test_text.append(text)

        self.test_data = [{'text': test_text[i], 'label': test_labels[i]} for i in range(len(test_text))]

        self.config = config

    
    def data(self):
        return self.train_data, self.test_data
    
