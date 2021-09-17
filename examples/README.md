# AzureML 2.0 CLI HuggingFace Fine-tuning Examples

## Prerequisites
1. If you don't have an Azure subscription, [create a free account](https://aka.ms/AMLFree).
2. Install and configure the [2.0 CLI machine learning extension](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli).
3. (Optional) Read over the documentation for [2.0 CLI jobs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli).

## Setup
1. Clone this repository and navigate to the examples directory:

```
git clone https://github.com/linydub/azureml-greenai-txtsum.git
cd azureml-greenai-txtsum/examples
```

2. Create a new Azure resource group and machine learning workspace and set defaults for resource group and workspace:

```bash
bash setup.sh
```

3. Create the workspace assets (environment, dataset, and compute target) used in the examples:

```bash
bash create-assets.sh
```

## Examples
Run a fine-tuning job defined through the example's YAML specification:
```
az ml job create --file ./jobs/pytorch-job.yml --web --stream
```
Job | Status | Description
--- | ------ | -----------
[jobs/pytorch-job.yml](jobs/pytorch-job.yml) | [![CLI Pytorch Job](../../../actions/workflows/cli-cmd-job-pytorch.yml/badge.svg)](../../../actions/workflows/cli-cmd-job-pytorch.yml) | Finetune an encoder-decoder Transformer model (BART) for dialogue summarization (SAMSum) with HuggingFace (PyTorch).
[jobs/deepspeed-job.yml](jobs/deepspeed-job.yml) | [![CLI DeepSpeed Job](../../../actions/workflows/cli-cmd-job-deepspeed.yml/badge.svg)](../../../actions/workflows/cli-cmd-job-deepspeed.yml) | Finetune an encoder-decoder Transformer model (BART) for dialogue summarization (SAMSum) with HuggingFace's DeepSpeed integration.
[jobs/sweep-job.yml](jobs/sweep-job.yml) | [![CLI Sweep Job](../../../actions/workflows/cli-sweep-job.yml/badge.svg)](../../../actions/workflows/cli-sweep-job.yml) | Hyperparameter tune an encoder-decoder Transformer model (BART) for dialogue summarization (SAMSum) with a grid search sweep job.

## Script usage (PyTorch)
[`jobs/src/main.py`](jobs/src/main.py) could be adapted or replaced with another script (e.g. [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py)) to fine-tune models for other NLP tasks.

*Script for text summarization was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py  

### New optional arguments
| Name | Type | Default | Description |
| :--- | :--- | :------ | :---------- |
| `dataset_path` | string | None | The path to the dataset files. Use this for loading data assets registered in the AzureML workspace. |
| `train_early_stopping` | bool | False | Whether to add EarlyStoppingCallback for training. This callback depends on `load_best_model_at_end` functionality to set best_metric in TrainerState. |
| `early_stopping_patience` | int | 1 | Use with `metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls. |
| `early_stopping_threshold` | float | 0.0 | Use with `metric_for_best_model` and `early_stopping_patience` to denote how much the specified metric must improve to satisfy early stopping conditions. |
| `freeze_embeds` | bool | False | Whether to freeze the model's embedding modules. |
| `freeze_encoder` | bool | False | Whether to freeze the model's encoder. |

## Known issues
*  

## Contents
| Directory | Description |
| --------- | ----------- |
| [`assets`](./assets) | Example workspace assets |
| [`jobs`](./jobs) | Example jobs for sample tasks |
| [`jobs/src`](./jobs/src) | Example training script and configs |

## References
AzureML:
- [Documentation](https://docs.microsoft.com/azure/machine-learning)
- [2.0 CLI examples](https://github.com/Azure/azureml-examples/tree/main/cli)
- [Private previews](https://github.com/Azure/azureml-previews)

HuggingFace:
- [Transformers documentation](https://huggingface.co/transformers/master/index.html)
- [Summarization example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization)

DeepSpeed:
- [JSON config documentation](https://www.deepspeed.ai/docs/config-json/)
