# AzureML 2.0 CLI HuggingFace Fine-tuning Example

WIP  

### Task Overview  

Datasets Info  

Models Info  


Prerequisites  

### Usage (PyTorch)

Includes FairScale, DeepSpeed, CodeCarbon, MLflow integrations.  

Adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py  

New Arguments:

- Parameter freezing
```
--freeze_embeds True
--freeze_encoder True
```
- Early stopping
```
--train_early_stopping True
--early_stopping_patience {int}
--early_stopping_threshold {float}
```

Job Examples:

- Pytorch distributed training

- DeepSpeed

- Sweep/Hyperdrive



### Example Benchmark Results

{benchmark_table}



*More benchmark details and comparisons in [HF model hub](https://huggingface.co/henryu-lin) and [benchmarks.md/experiments.md](https://github.com/henryu-lin/aml-txtsum/tree/main/research_project/incomplete)
