from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import zip_longest
from tqdm import tqdm
from dppy.finite_dpps import FiniteDPP
import torch
import openai
import backoff
import argparse
import numpy as np
import pandas as pd
import random
import json
import requests
import sys 
sys.path.append("..")
from data.sst2.SST2 import SST2

rng = np.random.RandomState(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_name', default='bert-base-nli-stsb-mean-tokens', type=str,
                   help='roberta-base, roberta-large, bart-base, bart-large, \
                   bert-base-nli-stsb-mean-tokens, bert-large-nli-stsb-mean-tokens, \
                   roberta-base-nli-stsb-mean-tokens, roberta-large-nli-mean-tokens, roberta-large-nli-stsb-mean-tokens')
parser.add_argument('--embed_type', default='CLS', type=str, help='CLS or mean')
parser.add_argument('--Q', default='text', help='sentence, source, table, text, q')
parser.add_argument('--A', default='label', help='label, target, sentence, code, a')
parser.add_argument('--reversed', action='store_true')
parser.add_argument('--label', default = 'gold', help = 'whether to flip labels or not', choices=['gold', 'random'])
parser.add_argument("--exec_model", default="text", help="gpt-3.5-turbo")
parser.add_argument('--use_hf_model', type = bool, default=False)
parser.add_argument("--api_key", default="text", required=True)
parser.add_argument("--label", default="gold", help="whether to flip labels or not")
parser.add_argument('--temp', type=float, default=0.7)
parser.add_argument('--min_num_neighbors', default=8, type=int)
parser.add_argument('--max_num_neighbors', default=8, type=int)
parser.add_argument('--num_choices', default=3, type=int)
parser.add_argument('--num_permutations', default=1, type=int)

args = parser.parse_args()

Q = args.Q
A = args.A

model = SentenceTransformer(args.encoder_name).to(device)


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
    

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
    
def pick_elements(indices, corpus):
    picked_elements = [corpus[int(i)] for i in indices]
    return picked_elements

def get_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    return model, tokenizer


def call_llama2(model, tokenizer, text: str, num_tokens: int = 10, temp=0.000001):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=num_tokens + inputs.input_ids.shape[1], do_sample=True, 
                        top_k=50, top_p=0.95, temperature=temp, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # remove the input text
    output = output[len(text):]
    return output.strip()


def encode(model, corpus):
    
    embeddings = []

    for corpus_tmp in tqdm(grouper(corpus, 1)):
        sentence_embeddings = model.encode(corpus_tmp)
        embeddings.append(sentence_embeddings)
            
    return np.concatenate(embeddings, axis=0)


sst2_train, sst2_test = SST2('sst2').data()
train_corpus = [example[Q] for example in sst2_train]
emb_train = torch.from_numpy(encode(model, train_corpus))
kernel_matrix = np.dot(emb_train, emb_train.T)
kernel_matrix = kernel_matrix / (np.linalg.norm(emb_train, axis=1)[:, np.newaxis] * np.linalg.norm(emb_train, axis=1))


dpp = FiniteDPP("likelihood", **{"L": kernel_matrix})

num2label = {"0": "negative", "1": "positive"}

if args.use_hf_model:
    
    API_URL = f"https://api-inference.huggingface.co/models/{args.exec_model}"
    headers = {"Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json"}
    tokenizer = AutoTokenizer.from_pretrained(args.exec_model)
    

else:
    
    openai.api_key = args.api_key
    openai.api_version = '2023-05-15'
    


  
for i in range(args.min_num_neighbors, args.max_num_neighbors+1):
    
    instructions = []
    ground_truth = []
    outputs = []
    examples = []
    
    for query in tqdm(sst2_test):
        
        q = query['text']
        label = num2label[str(query['label'])]
        
        train_indices = dpp.sample_mcmc_k_dpp(size = i, random_state=np.random.RandomState(np.random.randint(0, 4000)))
        train_ins = pick_elements(train_indices, sst2_train)
        
        for _ in range(args.num_permutations):
            
            random.shuffle(train_ins)       
            permutation = train_ins
            
            ice_examples = ""
            for doc in permutation:
                ice_examples += f"Input:{doc['text']}\nOutput:{num2label[doc['label']]}\n\n"                         
            prompt = f"{ice_examples}\nInput:{q}\nOutput:"      
            
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

    df.to_csv('./outputs/{}_sst2_dpp.csv'.format(args.exec_model))




