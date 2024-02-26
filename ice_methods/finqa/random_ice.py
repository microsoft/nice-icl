from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from glob import glob
import argparse
import torch
import json
import requests
import openai
import backoff
import pandas as pd
import numpy as np
import sys, os
import random
sys.path.append("..")
from data.finqa.FinQA import FinQA
root_dir = os.path.dirname(os.getcwd())

MAX_INT = sys.maxsize

device = torch.device('cuda')

prompts = []
for file in glob(os.path.join(root_dir,'instructions/finqa/*.txt')):
    
    ins_name = file.split("/")[-1][:-4]
    ins = open(file, 'r').read().strip()
    prompts.append({ins_name: ins})


parser = argparse.ArgumentParser()
parser.add_argument('--Q', default='text', help='sentence, source, table, text, q')
parser.add_argument('--A', default='label', help='label, target, sentence, code, a')
parser.add_argument('--label', default = 'gold', help = 'whether to flip labels or not', choices=['gold', 'random'])
parser.add_argument("--exec_model", default="text", help="gpt-3.5-turbo")
parser.add_argument('--use_hf_model', type = bool, default=False)
parser.add_argument("--api_key", default="text", required=True)
parser.add_argument('--temp', type=float, default=0.7)
parser.add_argument('--min_num_neighbors', default=8, type=int)
parser.add_argument('--max_num_neighbors', default=8, type=int)
parser.add_argument('--num_choices', default=3, type=int)
parser.add_argument('--num_permutations', default=1, type=int)

args = parser.parse_args()


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
@backoff.on_exception(backoff.expo, openai.error.APIConnectionError)
@backoff.on_exception(backoff.expo, openai.error.Timeout)
def completions_with_backoff(**kwargs):
  try:
    response = openai.ChatCompletion.create(**kwargs)
  except openai.error.InvalidRequestError as e:
    response = ""
    return response
  if'content' in response["choices"][0].message.keys():
    return response["choices"][0].message.content.strip()
  else:
    return ""


def query_request(payload, kwargs):
    json_body = {
        "inputs": f"{payload}",
                "parameters": kwargs
        }
    data = json.dumps(json_body)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    try:
        return json.loads(response.content.decode("utf-8"))
    except:
        return response
  
    
def get_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    return model, tokenizer


def call_llama2(model, tokenizer, text: str, num_tokens: int = 1, temp=0.000001):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=num_tokens + inputs.input_ids.shape[1], do_sample=True, 
                        top_k=50, top_p=0.95, temperature=temp, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    output = output[len(text):]
    return output.strip()
    

def shuffle_labels(data_list, field_name):
    # Extract the values of the specified field from each dictionary
    field_values = [d[field_name] for d in data_list]

    # Shuffle the field values
    random.shuffle(field_values)

    # Update the dictionaries with the shuffled values
    for i, d in enumerate(data_list):
        d[field_name] = field_values[i]
        
    return data_list


FinQA_train, FinQA_test = FinQA().data()

if args.use_hf_model:
    
    API_URL = f"https://api-inference.huggingface.co/models/{args.exec_model}"
    headers = {"Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json"}
    tokenizer = AutoTokenizer.from_pretrained(args.exec_model)
    

else:
    
    openai.api_key = args.api_key
    openai.api_version = '2023-05-15'
    

for p_dict in prompts:
  
    p_name, p = next(iter(p_dict.items()))
    
    for i in range(args.min_num_neighbors, args.max_num_neighbors+1):
      
        instructions = []
        ground_truth = []
        outputs = []
        examples = []


        for query in tqdm(FinQA_test):
            
            q = query['text']
            label = query['label']

            for _ in range(args.num_choices):
            
                retrieved_documents = np.random.choice(FinQA_train, size=i, replace=True)
            
                for _ in range(args.num_permutations):
              
                    random.shuffle(retrieved_documents)
                    permutation = retrieved_documents
            
                    if args.label == 'random': 
                        permutation = shuffle_labels(permutation, 'label')
            
                    ice_examples = "\n\n"
                    for doc in permutation:
                        ice_examples += f"Input:{doc['text']}\nOutput:{doc['label']}\n\n"
                    
                    prompt = f"{p}{ice_examples}Now answer the below query. Do not output anything else other than the final answer.\n\nInput:{q}\nOutput:"
                    
                    messages = [{"role": "system", "content" : "You are a kind and helpful assistant."}, {"role": "user", "content": prompt}]
                    
                    if args.use_hf_model:
                        
                        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
                        prompt = tokenizer.decode(tokenized_chat[0])
                        response  = query_request(prompt, {"top_p":0.9, "temperature":args.temp})
                    
                        try:
                            prediction = response[0]['generated_text'].strip()
                        except Exception as e:
                            prediction =  ''
            
                    else:
                        
                        prediction = completions_with_backoff(model= args.exec_model, messages=messages, temperature=args.temp) 
                        
                    instructions.append(prompt)
                    ground_truth.append(label)
                    outputs.append(prediction)
                    examples.append(permutation)
              
        res_dict = {"instructions": instructions, "answers": ground_truth, "output": outputs, "examples": examples}
        df = pd.DataFrame.from_dict(res_dict)

        p_name = f"{i}s_{p_name}"
        df.to_csv('./outputs/{}_finqa_random_{}_{}.csv'.format(args.exce_model, p_name, args.label))
                  
              


