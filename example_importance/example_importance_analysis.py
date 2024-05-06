from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import os, sys, time
import openai
from pprint import pprint
import pandas as pd
import random as rd
import argparse
import itertools as it
import backoff
import json
import requests
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join('..')))
from data.sst5.SST5 import SST5
from data.sst2.SST2 import SST2
from data.nl2bash.NL2BASH import NL2BASH
from data.Break.Break import Break
from data.mtop.MTOP import MTOP
from data.trec.TREC import TREC
from data.mnli.MNLI import MNLI
from data.finqa.FinQA import FinQA
from data.medqa.MedQA import MedQA

device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name()


parser = argparse.ArgumentParser()
parser.add_argument('--exec_model', default="gpt-4")
parser.add_argument('--use_hf_model', type = bool, default=False)
parser.add_argument('--ins', type = bool, default=False)
parser.add_argument('--ins_type', type = str, default='task-definition')
parser.add_argument('--embed_model', default="gtr-t5-xxl")
parser.add_argument('--api_key',required=True)
parser.add_argument('--task', default="sst5")
parser.add_argument('--num_queries', type=int, default=50)
parser.add_argument('--num_bins', type=int, default=10)
parser.add_argument('--num_shots', type=int, default=4)
parser.add_argument('--temp', type=float, default=0.0001)
args = parser.parse_args()

def get_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True, 
    quantization_config=quantization_config
    )
    
    return model, tokenizer

def call_llama2(model, tokenizer, text: str, temp=0.000001):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    generate_ids = model.generate(inputs.input_ids, do_sample=True, 
                        top_k=50, top_p=0.95, temperature=temp, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # remove the input text
    print("Original Text:", output)
    output = output[len(text):]  
    return output.strip()


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
    
    
embed_model = SentenceTransformer(args.embed_model)


@backoff.on_exception(backoff.expo, openai.error.APIError)
@backoff.on_exception(backoff.expo, openai.error.RateLimitError) 
@backoff.on_exception(backoff.expo, openai.error.APIConnectionError)
def completions_with_backoff(**kwargs):
  try:
    response = openai.ChatCompletion.create(**kwargs)
  except openai.error.InvalidRequestError as e:
    response = ""
    return response
    
  if 'content' not in response["choices"][0].message.keys():
    return ""
  return response["choices"][0].message.content.strip()

# code for importance analysis

class Queries:
    
    def __init__(self, test_data):
        test_x = [z['text'] for z in test_data]
        test_y = [z['label'] for z in test_data]
        self.queries = []
        self.ground_truth = []
        self.embeddings = []
        
        # pick random examples from the dataset
        for i in tqdm(rd.sample(range(len(test_x)),args.num_queries)):
            self.queries.append(test_x[i])
            self.ground_truth.append(test_y[i])

        self.embeddings = embed_model.encode(self.queries)
            
        self.sorted_examples = [
            [] for _ in range(len(self.queries))
        ]
        self.bins = [
            [] for _ in range(len(self.queries))
        ]

if args.task == 'sst5':
    
    sst5 = SST5('sst5', {})
    sst5_train, sst5_test = sst5.data()
    sst5_queries = Queries(sst5_test)
    
if args.task == 'sst2':

    sst2 = SST2('sst2', {})
    sst2_train, sst2_test = sst2.data()
    sst2_queries = Queries(sst2_test)
    
if args.task == 'break':
    
    break_data = Break('break')
    break_data_train, break_data_test = break_data.data()
    break_queries = Queries(break_data_test)
    
if args.task == 'mtop':
    
    mtop_data = MTOP('MTOP')
    mtop_data_train, mtop_data_test = mtop_data.data()
    mtop_queries = Queries(mtop_data_test)
    
if args.task == 'trec':
    
    trec_data = TREC('TREC', {})
    trec_data_train, trec_data_test = trec_data.data()
    trec_queries = Queries(trec_data_test)
    
if args.task == 'mnli':
    
    mnli_data = MNLI('MNLI', {})
    mnli_data_train, mnli_data_test = mnli_data.data() 
    mnli_queries = Queries(mnli_data_test)
    
if args.task == 'nl2bash':
    
    nl2bash_data = NL2BASH('NL2BASH', {})
    nl2bash_data_train, nl2bash_data_test = nl2bash_data.data()
    nl2bash_queries = Queries(nl2bash_data_test)
    
if args.task == 'finqa':
    
    finqa_data = FinQA()
    finqa_data_train, finqa_data_test = finqa_data.data()
    finqa_queries = Queries(finqa_data_test)
    
if args.task == 'medqa':
    
    medqa_data = MedQA('MedQA', {})
    medqa_data_train, medqa_data_test = medqa_data.data()
    medqa_queries = Queries(medqa_data_test)

def fill_sorted_examples(queries_class, train_data):
    # first, embed all train examples
    train_x = [z['text'] for z in train_data]
    train_y = [z['label'] for z in train_data]
    
    train_embed = embed_model.encode(train_x) 
    # for each query, sort the train examples by cosine similarity
    for i, query in enumerate(queries_class.queries):
        query_embed = queries_class.embeddings[i]
        # compute cosine similarity
        cos_sim = np.dot(train_embed, query_embed) / (np.linalg.norm(train_embed, axis=1) * np.linalg.norm(query_embed))
        # sort by cosine similarity
        sorted_examples_with_sim = sorted(zip(train_x, train_y, cos_sim), key=lambda x: x[2], reverse=True)
        queries_class.sorted_examples[i] = sorted_examples_with_sim
        

if args.task == 'sst5':
    fill_sorted_examples(sst5_queries, sst5_train)
    
if args.task == 'sst2':
    fill_sorted_examples(sst2_queries, sst2_train)
    
if args.task == 'break':
    fill_sorted_examples(break_queries, break_data_train)

if args.task == 'mtop':
    fill_sorted_examples(mtop_queries, mtop_data_train)
    
if args.task == 'trec':
    fill_sorted_examples(trec_queries, trec_data_train)
    
if args.task == 'mnli':
    fill_sorted_examples(mnli_queries, mnli_data_train)
    
if args.task == 'nl2bash':
    fill_sorted_examples(nl2bash_queries, nl2bash_data_train)
    
if args.task == 'finqa':
    fill_sorted_examples(finqa_queries, finqa_data_train)
    
if args.task == 'medqa':
    fill_sorted_examples(medqa_queries, medqa_data_train)


def fill_bins(queries_class):
    # clear bins
    queries_class.bins = [
        [] for _ in range(len(queries_class.queries))
    ]
    # fill bins of 10% of the train size from the sorted examples list for each query
    for i, query in enumerate(queries_class.queries):
        sorted_examples_with_sim = queries_class.sorted_examples[i]
        # fill bins
        factor = len(sorted_examples_with_sim) // args.num_bins
        for j in range(args.num_bins):
            start = j * factor
            end = (j + 1) * factor
            # add only text to the bins
            queries_class.bins[i].append([(x[0], x[1]) for x in sorted_examples_with_sim[start:end]])

if args.task == 'sst5':
    fill_bins(sst5_queries)
if args.task == 'sst2':
    fill_bins(sst2_queries)
if args.task == 'break':
    fill_bins(break_queries)
if args.task == 'mtop':
    fill_bins(mtop_queries)
if args.task == 'trec':
    fill_bins(trec_queries)
if args.task == 'mnli':
    fill_bins(mnli_queries)
if args.task == 'nl2bash':
    fill_bins(nl2bash_queries)
if args.task == 'finqa':
    fill_bins(finqa_queries)
if args.task == 'medqa':
    fill_bins(medqa_queries)
    
print("Embeddings done")


if args.use_hf_model:
    
    API_URL = f"https://api-inference.huggingface.co/models/{args.exec_model}"
    headers = {"Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json"}
    tokenizer = AutoTokenizer.from_pretrained(args.exec_model)
    

else:
    
    openai.api_key = args.api_key
    openai.api_version = '2023-05-15'

    
root_dir = os.path.dirname(os.getcwd())  
if args.ins:
    if args.task in ['sst2', 'sst5', 'mnli', 'trec']:
        instruction = open(os.path.join(root_dir, f'instructions/{args.task}/{args.ins_type}.txt'), 'r').read().strip()
    else:
        instruction = open(os.path.join(root_dir, f'instructions/{args.task}/{args.ins_type}.txt'), 'r').read().strip()
    
else:
    instruction = open(os.path.join(root_dir, f'instructions/{args.task}/no-instruction.txt'), 'r').read().strip()  


        
        
    
def run_analysis(queries_class, path):
    # first, create a dataframe with the following columns: query, bin, examples, ground_truth, prediction
    df = pd.DataFrame(columns=['query', 'bin', 'examples', 'ground_truth', 'prediction'])
    # for each query, fill the dataframe
    for i, query in tqdm(enumerate(queries_class.queries)):
        print(f'Query {i}')
        # for each bin, fill the dataframe
        messages = [
 {"role": "system", "content" : "You are a kind and helpful assistant."}
]
        for j, bin_examples in enumerate(queries_class.bins[i]):
            # loop 10 times
            for k in range(10):
                # sample 4 random examples from the bin
                random_examples = rd.sample(bin_examples, args.num_shots)
                simple_instruction = f"{instruction}\n\n"
                prompt = simple_instruction
                                
                if args.task == 'sst2':
                    
                    num2label = {"0": "negative", "1": "positive"}
                    for random_example in random_examples:
                        prompt += f'Input:{random_example[0]}\nOutput:{num2label[str(random_example[1])]}\n\n'
                        
                    prompt += f'Now answer the below query. Do not output anything else other than the final answer.\n\nInput:{query}\nOutput:'

                    messages = [{"role": "system", "content" : "You are a kind and helpful assistant."}, {"role": "user", "content": prompt}]
                    
                    if args.use_hf_model:
                        
                        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
                        prompt = tokenizer.decode(tokenized_chat[0])
                        response  = query_request(prompt, {"top_p":0.9, "temperature":args.temp})
                    
                        try:
                            prediction = response[0]['generated_text'][len(prompt):].strip()
                        except Exception as e:
                            prediction =  ''
            
                    else:
                        
                        prediction = completions_with_backoff(model= args.exec_model, messages=messages, temperature=args.temp) 
                        
                    df.loc[len(df)] = [query, j, random_examples, num2label[str(queries_class.ground_truth[i])], prediction]

                        
                elif args.task == 'sst5':
                    
                    num2label = {"0": "very negative", "1": "negative", "2": "neutral", "3": "positive", "4": "very positive"}
                    for random_example in random_examples:
                        prompt += f'Input:{random_example[0]}\nOutput:{num2label[str(random_example[1])]}\n\n'
    
                    prompt += f'Now answer the below query. Do not output anything else other than the final answer.\n\nInput:{query}\nOutput:'
                    
                    messages = [{"role": "system", "content" : "You are a kind and helpful assistant."}, {"role": "user", "content": prompt}]
                    
                    if args.use_hf_model:
                        
                        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
                        prompt = tokenizer.decode(tokenized_chat[0])
                        response  = query_request(prompt, {"top_p":0.9, "temperature":args.temp})
                    
                        try:
                            prediction = response[0]['generated_text'][len(prompt):].strip()
                        except Exception as e:
                            prediction =  ''
            
                    else:
                        
                        prediction = completions_with_backoff(model= args.exec_model, messages=messages, temperature=args.temp) 
                        
                    df.loc[len(df)] = [query, j, random_examples, num2label[str(queries_class.ground_truth[i])], prediction]
                    
                        
                elif args.task == 'mnli':
                    
                    num2label = {"0": "entailment", "1": "neutral", "2": "contradiction"}
                    for random_example in random_examples:
                        prompt += f'Input:{random_example[0]}\nOutput:{num2label[str(random_example[1])]}\n\n'
                        
                    prompt += f'Now answer the below query. Do not output anything else other than the final answer.\n\nInput:{query}\nOutput:'
                    
                    messages = [{"role": "system", "content" : "You are a kind and helpful assistant."}, {"role": "user", "content": prompt}]
                    
                    if args.use_hf_model:
                        
                        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
                        prompt = tokenizer.decode(tokenized_chat[0])
                        response  = query_request(prompt, {"top_p":0.9, "temperature":args.temp})
                    
                        try:
                            prediction = response[0]['generated_text'][len(prompt):].strip()
                        except Exception as e:
                            prediction =  ''
            
                    else:
                        
                        prediction = completions_with_backoff(model= args.exec_model, messages=messages, temperature=args.temp) 
                        
                    df.loc[len(df)] = [query, j, random_examples, num2label[str(queries_class.ground_truth[i])], prediction]
 
            
                else: 
                     

                    for random_example in random_examples:           
                        prompt += f'Input:{random_example[0]}\nOutput:{random_example[1]}\n\n'
                    
                    prompt += f'Now answer the below query. Do not output anything else other than the final answer.\n\nInput:{query}\nOutput:'
                    
                    messages = [{"role": "system", "content" : "You are a kind and helpful assistant."}, {"role": "user", "content": prompt}]
                    
                    if args.use_hf_model:
                        
                        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
                        prompt = tokenizer.decode(tokenized_chat[0])
                        response  = query_request(prompt, {"top_p":0.9, "temperature":args.temp})
                    
                        try:
                            prediction = response[0]['generated_text'][len(prompt):].strip()
                        except Exception as e:
                            prediction =  ''
            
                    else:
                        
                        prediction = completions_with_backoff(model= args.exec_model, messages=messages, temperature=args.temp) 
                        
                    df.loc[len(df)] = [query, j, random_examples, queries_class.ground_truth[i], prediction]

        
    df.to_csv(path, index=False)
    
    
if args.task == 'sst5':
    if args.ins:    
        run_analysis(sst5_queries, './example_importance/sst5_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(sst5_queries, './example_importance/sst5_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))


if args.task == 'mtop':
    if args.ins:    
        run_analysis(mtop_queries, './example_importance/mtop_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(mtop_queries, './example_importance/mtop_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))
    

if args.task == 'break':
    if args.ins:    
        run_analysis(break_queries, './example_importance/break_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(break_queries, './example_importance/break_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))
   
    
if args.task == 'sst2':
    if args.ins:    
        run_analysis(sst2_queries, './example_importance/sst2_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(sst2_queries, './example_importance/sst2_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))
    
    
if args.task == 'trec':
    if args.ins:    
        run_analysis(trec_queries, './example_importance/trec_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(trec_queries, './example_importance/trec_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))
    
    
if args.task == 'mnli':
    if args.ins:    
        run_analysis(mnli_queries, './example_importance/mnli_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(mnli_queries, './example_importance/mnli_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))
                                                                                               
if args.task == 'nl2bash':
    if args.ins:    
        run_analysis(nl2bash_queries, './example_importance/nl2bash_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(nl2bash_queries, './example_importance/nl2bash_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))
            
if args.task == 'finqa':
    if args.ins:    
        run_analysis(finqa_queries, './example_importance/finqa_{}_gtr_t5_xxl_with-ins_1.csv'.format(args.exec_model.split('/')[-1]))
    else:
        run_analysis(finqa_queries, './example_importance/finqa_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))
    
if args.task == 'medqa':
    if args.ins:    
        run_analysis(medqa_queries, './example_importance/medqa_{}_gtr_t5_xxl_with-ins_{}.csv'.format(args.exec_model.split('/')[-1], args.ins_type))
    else:
        run_analysis(medqa_queries, './example_importance/medqa_{}_gtr_t5_xxl_no-ins.csv'.format(args.exec_model.split('/')[-1]))