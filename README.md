**NICE:** Our research challenges the prevailing notion that optimizing in-context examples (ICE) consistently boosts language model performance. We demonstrate that for tasks with detailed instructions, ICE optimization may yield diminishing returns, proposing a task-specific metric (NICE) to guide when to prioritize ICE optimization over instruction optimization in new tasks.

[Paper Link](https://arxiv.org/abs/2402.06733)

## Dependencies
```
pip install requirements.txt
```
## Experiments
Instructions to run different experiments described in the paper
### Example Importance Analysis
```
$ cd example_importance
```
If you are using a Huggingface model via API access, then run the following command, with instructions:
```
python example_importance_analysis.py --task [TASK_NAME] --exec_model [HF-MODEL_NAME] --use_hf_model True --api_key [HF-API_KEY] --ins True --ins_type [INSTRUCTION_TYPE]
```
If you are using a Huggingface model via API access, then run the following command, without instructions:
```
python example_importance_analysis.py --task [TASK_NAME] --exec_model [HF-MODEL_NAME] --use_hf_model True --api_key [HF-API_KEY]
```
If you are using a OpenAI model via API access, then run the following command, with instructions:
```
python example_importance_analysis.py --task [TASK_NAME] --exec_model [OPENAI-MODEL_NAME]  --api_key [OPENAI-API_KEY] --ins True --ins_type [INSTRUCTION_TYPE]
```
If you are using a OpenAI model via API access, then run the following command, without instructions:
```
python example_importance_analysis.py --task [TASK_NAME] --exec_model [OPENAI-MODEL_NAME]  --api_key [OPENAI-API_KEY]
```

### Comparison with ICE selection methods
```
$ cd ice_methods/[TASK_NAME]
```
BM25 Retriever
```
python bm25.py
```
Topk-BERT 
```
python topk_bert.py
```
Topk-DPP
```
python topk_dpp.py
````
Random
```
python random_ice.py
```

### Label Perturbation with/without Instructions
```
$ cd ice_methods/[TASK_NAME]
```
With Gold Labels:
```
python random_ice.py --label gold
```
With Random Labels:
```
python random_ice.py --label random
```






