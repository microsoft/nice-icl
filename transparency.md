# Transparency.md for NICE-LLM

## Overview
To optimize in-context examples or not? Given a task and a target LLM, we provide an efficient metric to decide whether task performance is sensitive to choice of in-context examples. 

Our algorithm requires a short task description and a set of example demonstrations for the task. Since it measures invariance to ICE, we call the metric  Normalized Invariance to Choice of Examples (NICE). 

## Objective
Optimizing the prompt for a task involves considerable effort and API cost.  The main objective of NICE is to provide guidance on which part of the prompt would optimization help the most: the in-context examples or the task instruction.  In particular, the NICE metric can be computed wrt. different levels of detail in the task instruction, to evaluate the extent of diminishing returns of optimizing ICE as task instruction is improved. 

Note that the NICE metric provides a heuristic. A NICE score near 1 for a task implies minimal impact of ICE choice on model performance; therefore a prompt with random ICE should perform at par with a prompt with optimized ICE. For convenience, we also provide a pre-computed list of NICE metric scores for popularly used benchmarks for ICE optimization. 

## Audience
The NICE-LLM metric is intended for researchers, AI practitioners, and industry professionals who are interested in the optimizing erformance of LLMs on specific tasks.

## Intended Uses
This project is primarily designed for research and experimental purposes. We strongly recommend conducting further testing and validation before considering its application in industrial or real-world scenarios.

The proposed NICE metric can be used to avoid costly ICE optimization algorithms when the task performance is insensitive to choice of examples in the prompt. For example, it can be used to decide whether ICE optimization is needed for classification tasks such as semantic similarity, question-answering tasks such as multiple choice questions, information retrieval tasks, and problem-solving tasks such as math problems.  


## Out of Scope Uses
NICE metric is not intended to be used to circumvent any policies adopted by LLM providers.

## Evaluation
We evaluated the NICE metric on a range of tasks including sentiment classification, natural language inference, question answering, bash command generation and semantic parsing. On these tasks, we found that the NICE metric is an efficient way to understand the relative benefit of optimizing in-context examples for LLMs like GPT-4 and GPT-3.5-turbo. In particular, the NICE metric correlates well with the obtained acccuracy difference between prompts with random and optimized examples for each task.
For more information, check out our ACL 2024 paper: https://arxiv.org/abs/2402.06733


## Limitations
- NICE metric was tested only on popular NLP benchmarks. Performance of the method on real-world datasets may differ.
- The accuracy of the NICE score depends on the diversity and representativeness of the examples provided for metric computation. 


## Feedback and Collaboration
We welcome feedback and collaboration from our audience. If you have suggestions, questions, or would like to contribute to the project, please feel free to contact us at amshar@microsoft.com.

## Future Updates
The NICE-LLM project is an ongoing initiative. If you would like to contribute the NICE score for new task, you can raise a Pull Request to update the tasks table.
