---
language: en
tags:
- azureml
- bart
- summarization
license: apache-2.0
datasets:
- samsum
model-index:
- name: bart-large-samsum
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
    Kevin: Yea, he said he'll be arriving a bit later at around 7 since he gets off of work at 6. Have you taken out the garbage yet?
    Henry: Oh I forgot. I'll do that once I'm finished with my assignment for my math class.
    Kevin: Yea, you should take it out as soon as possible. And also, Nate is bringing his girlfriend.
    Henry: Nice, I'm really looking forward to seeing them again.
---

## `bart-large-samsum`

This model was trained using Microsoft's `AzureML`. It was fine-tuned on the `SAMSum` corpus from `facebook/bart-large` checkpoint.

More information on the fine-tuning process (includes samples and benchmarks):  
https://github.com/henryu-lin/aml-txtsum  
*(currently still WIP, major updates coming soon: 7/6/21~7/9/21)*

## Resource Usage
These results are retrieved from AzureML Studio's resource monitoring module. All experiments were ran on AzureML's low priority clusters.

| key | value |
| --- | ----- |
| AzureML SKU | ND40rs_v2 (8 X V100 32GB) |
| Region | US West 2 |
| Run Duration | 5m 26s |
| Compute Cost (LowPriority/Dedicated) | $0.40/$2.00 (USD) |
| Average CPU Utilization | 45.7% |
| Average GPU Utilization | 73.3% |
| GPU Memory Usage (Avg/Peak) | 25.44/29.27 (GB) |
| Total GPU Energy Usage | 354.17 (kJ) |

*Compute cost is calculated from run duration and SKU's price per hour. Updated SKU pricing could be found here: https://azure.microsoft.com/en-us/pricing/details/machine-learning/  
*Peak memory usage is calculated from average peak across all utilized GPUs.  

### Carbon Emissions
These results are obtained using `codecarbon`. The carbon emission is estimated from training runtime only (excluding setup and evaluation runtime).  
CodeCarbon: https://github.com/mlco2/codecarbon  

| key | value |
| --- | ----- |
| timestamp | 2021-07-07T01:51:45 |
| duration | 203.5628867149353 |
| emissions | 0.02357999921657574 |
| energy_consumed | 0.07923386833526794 |
| country_name | USA |
| region | Washington |
| cloud_provider | azure |
| cloud_region | westus2 |

## Hyperparameters
```yaml
fp16: True
per device batch size: 16
effective batch size: 128
epoch: 3.0
learning rate: 5e-5
weight decay: 0.1
seed: 1
```

## Usage
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="henryu-lin/bart-large-samsum")

conversation = '''Henry: Hey, is Nate coming over to watch the movie tonight?
    Kevin: Yea, he said he'll be arriving a bit later at around 7 since he gets off of work at 6. Have you taken out the garbage yet?
    Henry: Oh I forgot. I'll do that once I'm finished with my assignment for my math class.
    Kevin: Yea, you should take it out as soon as possible. And also, Nate is bringing his girlfriend.
    Henry: Nice, I'm really looking forward to seeing them again.
'''
summarizer(conversation)
```

## Results
| ROUGE | Score |
| ----- | ----- |
| eval_rouge1 | 55.2131 |
| eval_rouge2 | 30.0843 |
| eval_rougeL | 45.343 |
| eval_rougeLsum | 50.6611 |
| predict_rouge1 | 53.5615 |
| predict_rouge2 | 28.3713 |
| predict_rougeL | 44.1337 |
| predict_rougeLsum | 49.3075 |

| Metric | Value |
| ------ | ----- |
| eval_gen_len | 30.8704 |
| predict_gen_len | 30.4969 |
| train_loss | 1.2001634356619297 |
| eval_loss | 1.4273128509521484 |
| predict_loss | 1.4450620412826538 |
| train_runtime | 203.4396 |
| train_samples | 14732 |
| train_samples_per_second | 217.244 |
| train_steps_per_second | 1.711 |
| eval_runtime | 23.7089 |
| eval_samples | 818 |
| eval_samples_per_second | 34.502 |
| eval_steps_per_second | 0.295 |
| predict_runtime | 26.7401 |
| predict_samples | 819 |
| predict_samples_per_second | 30.628 |
| predict_steps_per_second | 0.262 |
| total_steps | 348 |
| total_flos | 4.26008990669865e+16 |