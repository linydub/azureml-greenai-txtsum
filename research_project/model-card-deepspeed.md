---
language: en
tags:
- azureml
- t5
- summarization
- deepspeed
license: apache-2.0
datasets:
- samsum
model-index:
- name: t5-3b-samsum-deepspeed
  results:
  - task: 
      name: Abstractive Text Summarization
      type: abstractive-text-summarization
    dataset:
      name: "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization" 
      type: samsum
widget:
- text: | 
    Henry: Hey, is Nate coming over to watch the movie tonight?
    Kevin: Yea, he said he'll be arriving a bit later at around 7 since he gets off of work at 6. Have you taken out the garbage yet? It's starting to make the kitchen really smell.
    Henry: Oh I forgot. I'll do that once I'm finished with my assignment for my math class.
    Kevin: Yea, you should take it out as soon as possible. And also, Nate is bringing his girlfriend too.
    Henry: Nice, I'm really looking forward to seeing them again.
---

## `t5-3b-samsum-deepspeed`
This model was trained using Microsoft's `AzureML` and `DeepSpeed`'s ZeRO 2 optimization. It was fine-tuned on the `SAMSum` corpus from `t5-3b` checkpoint.

More information on the fine-tuning process (includes samples and benchmarks):  
*(currently still WIP, major updates coming soon: 7/6/21~7/9/21)*

## Resource Usage
These results are retrieved from AzureML Studio's resource monitoring module. All experiments were ran on AzureML's low priority clusters.

| key | value |
| --- | ----- |
| AzureML SKU | ND40rs_v2 (8 X V100 32GB) |
| Region | US West 2 |
| Run Duration | 43m 51.05s |
| Compute Cost (LowPriority/Dedicated) | $3.22/$16.10 (USD) |
| Average CPU Utilization | 46.0% |
| Average GPU Utilization | 56.9% |
| GPU Memory Usage (Avg/Peak) | 26.77/30.49 (GB) |
| Total GPU Energy Usage | 2448.69 (kJ) |

*Compute cost is calculated from run duration and SKU's price per hour. Updated SKU pricing could be found here: https://azure.microsoft.com/en-us/pricing/details/machine-learning/  
*Peak memory usage is calculated from average peak across all utilized GPUs.  

### Carbon Emissions
These results are obtained using `codecarbon`. The carbon emission is estimated from training runtime only (excluding setup and evaluation runtime).  
CodeCarbon: https://github.com/mlco2/codecarbon  

| key | value |
| --- | ----- |
| timestamp | 2021-07-06T21:57:39 |
| duration | 1841.4621863365173 |
| emissions | 0.17802492531467784 |
| energy_consumed | 0.5982020339874927 |
| country_name | USA |
| region | Washington |
| cloud_provider | azure |
| cloud_region | westus2 |

## Hyperparameters
```yaml
fp16: True
per device batch size: 2
effective batch size: 16
epoch: 3.0
learning rate: 3e-5
weight decay: 0.0
seed: 1
```
*Same `per device batch size` for evaluations

### DeepSpeed
Optimizer = `AdamW`, Scheduler = `WarmupDecayLR`, Offload = `none`
```json
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 1000000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 1000000000,
    "contiguous_gradients": true
  }
```

## Usage
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="henryu-lin/t5-3b-samsum-deepspeed")

conversation = '''Henry: Hey, is Nate coming over to watch the movie tonight?
    Kevin: Yea, he said he'll be arriving a bit later at around 7 since he gets off of work at 6. Have you taken out the garbage yet? It's starting to make the kitchen really smell.
    Henry: Oh I forgot. I'll do that once I'm finished with my assignment for my math class.
    Kevin: Yea, you should take it out as soon as possible. And also, Nate is bringing his girlfriend too.
    Henry: Nice, I'm really looking forward to seeing them again.
'''
summarizer(conversation)
```

## Results
| ROUGE | Score |
| ----- | ----- |
| eval_rouge1 | 54.7875 |
| eval_rouge2 | 30.565 |
| eval_rougeL | 45.7625 |
| eval_rougeLsum | 50.3915 |
| predict_rouge1 | 53.6628 |
| predict_rouge2 | 29.0196 |
| predict_rougeL | 45.1257 |
| predict_rougeLsum | 49.171 |

| Metric | Value |
| ------ | ----- |
| eval_gen_len | 25.3399 |
| predict_gen_len | 24.9133 |
| train_loss | 1.1206104169494209 |
| eval_loss | 1.0732421875 |
| predict_loss | 1.087890625 |
| train_runtime | 1841.3751 |
| train_samples | 14732 |
| train_samples_per_second | 24.002 |
| train_steps_per_second | 1.501 |
| eval_runtime | 163.8357 |
| eval_samples | 818 |
| eval_samples_per_second | 4.993 |
| eval_steps_per_second | 0.317 |
| predict_runtime | 168.8245 |
| predict_samples | 819 |
| predict_samples_per_second | 4.851 |
| predict_steps_per_second | 0.308 |
| total_steps | 2763 |
| total_flos | 1.84452086400811e+17 |
