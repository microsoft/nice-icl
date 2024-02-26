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

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.