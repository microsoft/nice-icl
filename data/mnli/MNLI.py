from datasets import load_dataset
import numpy as np

class MNLI:

    def __init__(self, name, config=None):

        self.name = name
        dataset_train = load_dataset("multi_nli", split="train")
        dataset_train = dataset_train if len(dataset_train)<=10000 else np.random.choice(dataset_train, size=10000, replace=False)
        dataset_test = load_dataset("multi_nli", split="validation_matched")
        dataset_test = dataset_test if len(dataset_test)<=1000 else np.random.choice(dataset_test, size=1000, replace=False)

        train_data_sentences = []
        train_data_labels = []
        train_data_genres = []
        for i in range(len(dataset_train)):
            sentences = f"premise: {dataset_train[i]['premise']}\nhypothesis: {dataset_train[i]['hypothesis']}"
            label = dataset_train[i]['label']
            train_data_sentences.append(sentences)
            train_data_labels.append(str(label))
            train_data_genres.append(dataset_train[i]['genre'])

        self.train_data = [{'text': train_data_sentences[i], 'label': train_data_labels[i], 'label_text': str(train_data_labels[i]), 'genre': train_data_genres[i]} for i in range(len(train_data_sentences))]

        test_data_sentences = []
        test_data_labels = []
        test_data_genres = []
        for i in range(len(dataset_test)):
            sentences = f"premise: {dataset_test[i]['premise']}\nhypothesis: {dataset_test[i]['hypothesis']}"
            label = dataset_test[i]['label']
            test_data_sentences.append(sentences)
            test_data_labels.append(str(label))
            test_data_genres.append(dataset_test[i]['genre'])

        self.test_data = [{'text': test_data_sentences[i], 'label': test_data_labels[i], 'label_text': str(test_data_labels[i]), 'genre': test_data_genres[i]} for i in range(len(test_data_sentences))]
        self.config = config

    
    def data(self):
        return self.train_data, self.test_data