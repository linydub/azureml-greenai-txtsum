---
language:
- en
license: apache-2.0
tags:
- summarization
- azureml
- azure
- codecarbon
- bart
datasets:
- samsum
metrics:
- rouge
model-index:
- name: bart-large-samsum
  results:
  - task: 
      name: Abstractive Text Summarization
      type: abstractive-text-summarization
    dataset:
      name: SAMSum Corpus 
      type: samsum
    metrics:
    - name: Validation ROUGE-1
      type: rouge-1
      value: 55.2131
    - name: Validation ROUGE-2
      type: rouge-2
      value: 30.0843
    - name: Validation ROUGE-L
      type: rouge-L
      value: 45.343
    - name: Validation ROUGE-Lsum
      type: rouge-Lsum
      value: 50.6611
    - name: Test ROUGE-1
      type: rouge-1
      value: 53.5615
    - name: Test ROUGE-2
      type: rouge-2
      value: 28.3713
    - name: Test ROUGE-L
      type: rouge-L
      value: 44.1337
    - name: Test ROUGE-Lsum
      type: rouge-Lsum
      value: 49.3075
widget:
- text: | 
    Henry: Hey, is Nate coming over to watch the movie tonight?
    Kevin: Yea, he said he'll be arriving a bit later at around 7 since he gets off of work at 6. Have you taken out the garbage yet?
    Henry: Oh I forgot. I'll do that once I'm finished with my assignment for my math class.
    Kevin: Yea, you should take it out as soon as possible. And also, Nate is bringing his girlfriend.
    Henry: Nice, I'm really looking forward to seeing them again.
---

## `bart-large-samsum`
This model was trained using Microsoft's [`Azure Machine Learning Service`](https://azure.microsoft.com/en-us/services/machine-learning). It was fine-tuned on the [`SAMSum Corpus`](https://huggingface.co/datasets/samsum) from [`facebook/bart-large`](https://huggingface.co/facebook/bart-large) checkpoint.

## Fine-tune on AzureML
[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Flinydub%2Fazureml-greenai-txtsum%2Fmain%2F.cloud%2Ftemplate-hub%2Fischool-greenai%2Farm-bart-large-samsum.json) [![Visualize](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/visualizebutton.svg?sanitize=true)](http://armviz.io/#/?load=https://raw.githubusercontent.com/linydub/azureml-greenai-txtsum/main/.cloud/template-hub/ischool-greenai/arm-bart-large-samsum.json)

**[PREVIEW]** More information about the fine-tuning process (includes samples and benchmarks):  
https://github.com/linydub/azureml-greenai-txtsum

## Usage (Inference)
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="ischool-greenai/bart-large-samsum")

input_text = '''
    Henry: Hey, is Nate coming over to watch the movie tonight?
    Kevin: Yea, he said he'll be arriving a bit later at around 7 since he gets off of work at 6. Have you taken out the garbage yet?
    Henry: Oh I forgot. I'll do that once I'm finished with my assignment for my math class.
    Kevin: Yea, you should take it out as soon as possible. And also, Nate is bringing his girlfriend.
    Henry: Nice, I'm really looking forward to seeing them again.
'''
summarizer(input_text)
```

## Resource Usage
These results were retrieved from [`Azure Monitor Metrics`](https://docs.microsoft.com/en-us/azure/azure-monitor/essentials/data-platform-metrics). All experiments were ran on AzureML low priority compute clusters.

| key | value |
| --- | ----- |
| Region | US West 2 |
| AzureML Compute SKU | STANDARD_ND40RS_V2 |
| Compute SKU GPU Device | 8 X V100 32GB |
| Compute Node Count | 1 |
| Run Duration | 8m 28s |
| Compute Cost (LowPriority/Dedicated) | $3.11 / $0.62 USD |
| Average CPU Utilization | 41.3% |
| Average GPU Utilization | 63.2% |
| Average GPU Memory Usage | 22.08 GB |
| Total GPU Energy Usage | 433.90 kJ |


*Compute cost ($) is estimated from the run duration, number of compute nodes utilized, and SKU's price per hour. Updated SKU pricing could be found [here](https://azure.microsoft.com/en-us/pricing/details/machine-learning).  

### Carbon Emissions
These results were obtained using [`CodeCarbon`](https://github.com/mlco2/codecarbon). The carbon emissions are estimated from training runtime only (excl. setup and evaluation runtimes).  

| key | value |
| --- | ----- |
| timestamp | 2021-06-04T15:22:57 |
| duration | 4.730183839797974 |
| emissions | 0.00022838667136857883 |
| energy_consumed | 0.0007674283312116224 |
| country_name | USA |
| region | Washington |
| cloud_provider | azure |
| cloud_region | westus2 |

## Hyperparameters

- max_source_length: 512
- max_target_length: 90
- fp16: True
- seed: 1
- per_device_train_batch_size: 16
- per_device_eval_batch_size: 16
- learning_rate: 5e-5
- weight_decay: 0.1



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
| epoch | 3.0 |
| eval_gen_len | 30.8704 |
| eval_loss | 1.4273128509521484 |
| eval_runtime | 23.7089 |
| eval_samples | 818 |
| eval_samples_per_second | 34.502 |
| eval_steps_per_second | 0.295 |
| predict_gen_len | 30.4969 |
| predict_loss | 1.4450620412826538 |
| predict_runtime | 26.7401 |
| predict_samples | 819 |
| predict_samples_per_second | 30.628 |
| predict_steps_per_second | 0.262 |
| train_loss | 1.2001634356619297 |
| train_runtime | 203.4396 |
| train_samples | 14732 |
| train_samples_per_second | 217.244 |
| train_steps_per_second | 1.711 |
| total_steps | 348 |
| total_flops | 4.904092764943155e+16 |
