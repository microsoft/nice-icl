import json
import numpy as np


def convert_to_linear(table):
    if not table:
        return ""
    
    # Initialize an empty result string
    result = ""

    # Iterate through each row in the table
    for row in table:
        # Iterate through each column in the row and concatenate the values with '|'
        result += " | ".join(str(cell) for cell in row) + "\n"

    return result

class FinQA:
    def __init__(self):
        
        with open('./finqa_data/train.json', 'r') as f:
            self.dataset_train = json.load(f)
            
        with open('./finqa_datadata/test.json', 'r') as f:
            self.dataset_test = json.load(f)
            
        self.train_data = []
        self.test_data = []
        for item in self.dataset_train:
            
            example = dict()
            table = item["table"]
            processed_table = convert_to_linear(table)
            
            question = item["qa"]["question"]
            answer = item["qa"]["answer"]
            
            model_inps = item["qa"]["model_input"]
            
            text = ""
            text += "\n".join(inp[1] for inp in model_inps)
            
            example['text'] = text + '\n\n' + processed_table + '\nQuestion:' + question
            example['label'] = answer
          
            self.train_data.append(example)
        
        self.train_data = self.train_data if len(self.train_data)<=1000 else np.random.choice(self.train_data, size=1000, replace=False)
            
        for item in self.dataset_test:
            
            example = dict()
            table = item["table"]
            processed_table = convert_to_linear(table)
            
            question = item["qa"]["question"]
            answer = item["qa"]["answer"]
            
            model_inps = item["qa"]["model_input"]
            
            text = ""
            text += "\n".join(inp[1] for inp in model_inps)
            
            example['text'] = text + '\n\n' + processed_table + '\nQuestion:' + question
            example['label'] = answer
            
            self.test_data.append(example)
            
    def data(self):
        return self.train_data, self.test_data
    
            
                
            
            