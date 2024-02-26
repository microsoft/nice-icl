from datasets import load_dataset
import numpy as np

class MTOP:

    def __init__(self, name, config=None):
        self.name = name
        dataset_train = load_dataset("tasksource/mtop", split='train')
        dataset_train = dataset_train if len(dataset_train)<=10000 else np.random.choice(dataset_train, size=10000, replace=False)
        dataset_test = load_dataset("tasksource/mtop", split='validation')
        dataset_test = dataset_test if len(dataset_test)<=1000 else np.random.choice(dataset_test, size=1000, replace=False)

        lang_map = {"th_XX": "Thai", "es_XX": "Spanish", "en_XX": "English", "de_XX": "German", "hi_XX": "Hindi", "fr_XX": "French"}
        
        self.train_query = []
        self.train_lf = []
        self.train_lang = []
        self.train_domain = []

        for i in range(len(dataset_train)):
            self.train_query.append(dataset_train[i]['question'])
            self.train_lf.append(dataset_train[i]['logical_form'])
            self.train_lang.append(lang_map[dataset_train[i]['lang']])
            self.train_domain.append(dataset_train[i]['domain'])
            
        self.train_data = [{'text': self.train_query[i], 'label': self.train_lf[i], 'lang': self.train_lang[i], 'domain': self.train_domain[i]} for i in range(len(self.train_query))]

        self.test_query = []
        self.test_lf = []
        self.test_lang = []
        self.test_domain = []

        for i in range(len(dataset_test)):
            self.test_query.append(dataset_test[i]['question'])
            self.test_lf.append(dataset_test[i]['logical_form'])
            self.test_lang.append(lang_map[dataset_test[i]['lang']])
            self.test_domain.append(dataset_test[i]['domain'])
            
        self.test_data = [{'text': self.test_query[i], 'label': self.test_lf[i], 'lang': self.test_lang[i], 'domain': self.test_domain[i]} for i in range(len(self.test_query))]

        self.config = config


    def data(self):

        return self.train_data, self.test_data
