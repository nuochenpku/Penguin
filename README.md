# Penguin: A benchmack for Natural Response Generation in Chinese Chinese Reading Comprehension

[**Tasks**](#task-description) | [**Dataset**](#dataset) | [**Checkpoints**](#Checkpoints) |
[**Paper**](https://arxiv.org/abs/2010.04898) |
[**Citation**](#citation) | [**License**](#license)

This repository contains resources for accessing the official benchmarks, codes and checkpoints of the paper:  ***Natural Response Generation for Chinese Reading Comprehension***.

We introduce Penguin, an end-to-end Chinese question answering dataset comprising of 200K question-passage-answer-response pairs. The goal of this dataset is to provide a challenging benchmark for end-to-end Chinese Machine Readinng Comprehension that includes the well-informed responses of each question. 
Penguin can facilate the research to build generative QA models in Chinese and provide a relatively large-scale training corpus for Chinese communities.
Please refer to our paper Natural Response Generation for Chinese Reading Comprehension.


## DataSet
Penguin, in the hopes of creating sophisticated  GRC models that can generate natural responses for practical QA situations. Considering constructing such a  dataset on a large scale is non-trivial, we initialize our dataset  from the current Chinese MRC dataset corpus to get raw passage-question-answer  triplets, including CMRC 2018, DuReader, and ReCo.

It is extremely difficult and expensive to ask the annotator to start from scratch and write a response $\mathcal{R}$ that makes sense and sounds natural for each < $\mathcal{P}$, $\mathcal{Q}$, $\mathcal{A}$ > triplet.
Therefore, we  generate informative and fluent responses via the following steps: (1) We first utilize a specific response generative model to generate initial responses for the above triplets; (2) Then, we use state-of-the-art semantic matching models alongside certain manually-created criteria to exclude samples of incoherence and semantic shift; (3) Last, we employ three professional annotators to rewrite and recheck undesired cases, therefore regulating data quality. As a result, we collect a sequence of 4-tuples: < $\mathcal{P}$, $\mathcal{Q}$, $\mathcal{A}$, $\mathcal{R}$ > in Penguin, where $\mathcal{R}$ is the labeled response. 

Concretely, we store our dataset in json files:


```
{
  "Passage": "鳊鱼一直备受人们喜爱，鳊鱼的传统做法是清蒸或者红烧。当然不排除可以烧汤，但是鉴于鳊鱼的肉质特色，不是很适合烧汤的",
  "Query": "鳊鱼可以炖汤吗?",
  "Answer": "可以",
  "Response": "鳊鱼可以用来煮汤，但是一般不推荐这么做."
  
}
```

### Download

| [**Train set**] |  [**Dev Set**] | [**Test Set**] | [**ALL**] |


## Checkpoints

### Results
Here we report automatic and huaman evaluations results of four baselines in our paper.

![](Results.png) 

### End-to-end ckpt Download

|Model |  Large| Base | 
| :----- | :-------------------:| :------------------: |
| T5 | T5-base  | T5-small | 
| BART | BART-Large  | BART-base | 
| Prompt-BART | [Prompt-BART-Large](https://hkustgz-my.sharepoint.com/:f:/g/personal/nchen022_connect_hkust-gz_edu_cn/EjrfcimBm01EnQUUdqkntfQB7Ox9FDaB9JvsfC9GC4N88w?e=trpxZd)  | -| 

### Two stage ckpt Download
|Model |  Large| Base | 
| :----- | :-------------------:| :------------------: |
| T5 | T5-base(Answerer, Responser)  | T5-small(Answerer, Responser) | 
| BART | BART-Large(Answerer, Responser)  | BART-base(Answerer, Responser) | 
| Prompt-BART | Prompt-BART-Large(Answerer, Responser)  | -| 

