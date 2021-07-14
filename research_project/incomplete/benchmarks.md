# Experiments
WIP

## Environment Details
- Platform
- Hardware/SKU
- 

{env_table}

## Datasets Details
Sample count and sequence length (BPE tokens).

| Dataset | Train | Val | Test | Source (words/tokens) | Target (words/tokens) |
| ------- | ----- | --- | ---- | :---: | :---: |
| CNN-DM | 287K	| 13K | 11K | 781/906 | 56/63 |
| XSUM | 204K | 11K	| 11K | 431/488 | 23/27 |
| SAMSum | 14.7K | 818 | 819 | 94/147 | 20/26 |

Reference:
https://paperswithcode.com/paper/hierarchical-learning-for-generation-with/review/?hl=30985  

### SAMSum (Dialogue summarization)
- SAMSum dataset similar to XSUM.
- Used XSUM hyperparameters when fine-tuning.
- 

## Models Details
{model_tables}


## Comparisons
**Core Experiments:**  
- ZeRO optimization: Reduces GPU memory usage, *increases speed
- ZeRO offload/infinity (CPU): Reduces GPU memory usage, reduces speed
- Population based tuning over grid search (Hyperparameter tuning): Less runtime & compute, *better accuracy
- Floating point format (FP16- automatic mixed precision): Reduces GPU memory usage, increases speed
- Sparse Attention Kernel: Reduces GPU memory usage, *affects accuracy


**Other Experiments:**
- Parameter freezing (embedding, encoder modules): Reduces GPU memory footprint, increases throughput, *affects accuracy
- Sortish sampling: Reduces GPU memory usage, *reduces accuracy
- Data Truncation (sequence length): Reduces GPU memory usage, reduces accuracy
- Early stopping: Reduces runtime, *reduces accuracy

### Hyperparameters
- Batch size: Increase memory usage, increase speed
- DS bucket sizes: Increase memory usage, increase speed
- 

Applicable in general cases, some have trade-offs, asterisks prefix = situational, accuracy refers to task performance  

## Optimization Methods
Some situational cases could have complex interactions between employed methods/parameters, other things could influence optimization method's effectiveness or task performance like the fine-tuning corpus and model size.

Trade-offs:  
{table}  
Some experiment benchmark results:  
{table}  

## Benchmark Results
- R-1/R-2/R-L 
- Loss
- Resource usage & carbon metrics


---


# (May) Case Study: Carbon Emission Evaluation of Pretrained SOTA NLP Transformers on Supervised Abstractive Text Summarization

Goals from the initial GreenAI document:

-   Develop a methodology for operational lifecycle analysis across a ML project that maps key efficiency/power metrics, cost ($), carbon, and accuracy.

-   Compare the cost/benefit for different baseline architectures.

-   Create public-facing samples to provide an easy starting point for customer POC, which can then be customized for specific scenarios.

## [Scope]

NLP Task: Abstractive text summarization (English)

Architecture: SOTA & efficient Transformers (encoder-decoder)

Training strategy: Supervised, fine-tune pretrained models (transfer learning)

*Prefix (asterisk) are additional items outside of core scope.

## [Metrics]

Reference platform/service: Azure (cloud/datacenter)

Cost (monetary):

-   Compute (dollars/hour): VM charged on a timed basis independent of resource usage

-   Other services (storage, *data prep): Azure Portal (resource group cost analysis page)

Quality (task performance):

-   Standard (accuracy & precision): ROUGE

-   *SummEval ([[2007.12626] SummEval: Re-evaluating Summarization Evaluation](https://arxiv.org/abs/2007.12626))\
    Code: <https://github.com/Yale-LILY/SummEval#evaluation-toolkit>

Speed:

-   Runtime (seconds): HuggingFace

-   Throughput (samples/second, tokens/second): HuggingFace

Memory:

-   RAM, VRAM (GBs): HuggingFace, AzureML Studio

-   Disk space/storage (GBs):

Compute:

-   GPU, CPU (utilization %): AzureML Studio (run's monitoring tab)

-   FPOs: HuggingFace

Carbon emission: CO₂eq (lbs)

Power: Watts

## [Sample]

End-to-end notebook: Basic guide using efficient strategies validated by experiment results

*Cloud endpoint example: Realtime showcase of model's inference carbon emission

Scope: Setup, development, deployment, *maintenance

-   AzureML 2.0 (CLI, Studio)

-   HuggingFace (PyTorch)

-   MLflow

-   DeepSpeed

-   ONNX Runtime (Training, Inference, *Quantization)

-   NVIDIA Triton

-   *[Experiment-Impact-Tracker](https://github.com/Breakend/experiment-impact-tracker)/[CodeCarbon](https://github.com/mlco2/codecarbon)

## [Replications (POC)]

Models are fine-tuned with hyperparameter values mentioned in their paper or otherwise with sensible default values. Replication results are produced for baselines and cost/carbon telemetry.

Models: Pegasus, Bart, T5, ProphetNET, *[UniLM](https://github.com/microsoft/unilm)

-   *Efficient (sparsity):  BigBird-Pegasus, LED

Datasets: XSUM, CNN/DailyMail, Reddit-TIFU, SAMSum, *(subreddit: finance, gaming)

-   *Long sequence: BigPatent

## [Experiments]

Experiment results are produced for cost/benefit analysis and telemetry.

-   Compare models of comparable size on public datasets

-   *Evaluate the effects of pretraining corpus and methods/objectives (transfer learning)

-   *Compare environments/hardwares/platforms

-   *Compare tools/frameworks/libraries/methods

-   Evaluate model cost/compute/memory/time/sample efficiency (benchmarks)

-   *Low-resource cases

-   *Specific cases (multi-document, long sequences, domain-specific, etc.)

Parameters:

-   Model: architecture, size (params/layers/heads), checkpoint

-   Data (pretrained & training corpus): quantity, *quality

-   Hardware: datacenter/*consumer, GPU/CPU/*TPU, distributed/single-node

-   GPU (T4, P100, V100, A100, *RTX 2000/3000s)

-   Tools/Frameworks/Libraries: PyTorch, DeepSpeed, ONNX Runtime, *Horovod, *FairScale, *TorchScript, *TensorFlow

-   Methods: early stopping, *distillation, *pruning, *ensemble

-   Floating-point format: FP32, FP16 (mixed precision), INT8 quantization

-   *Optimizer: AdamW, Adafactor

-   *Hyperparam tuning: Bayesian optimization, grid/random search, PBT, etc.

Findings:

-   Training/development tips and insights

-   Problems and/or difficulties during development/iterative process

-   Empirical results (tables/graphs) & source code (github repo)

-   Trade-off evaluations

## [Research]

Literature review (March/April):

[Text Summarization Research](https://docs.google.com/document/d/1NYVLTFWWFR3RrBWE7PD8VyLcs4S7CHncFivojKF7A60/edit)

[Pretrained Models (TXTSUM)](https://docs.google.com/document/d/1DpBgdQ15dtYmVflVYpo7FKhvhtRIQUDSn_esZZBbEuc/edit#)

[PEGASUS evaluation time & cost estimates (AzureML)](https://docs.google.com/spreadsheets/d/1OJGhG2sBqxg29nEuawFtecL7y-3XPAM27Wb9E7TnZEE/edit#gid=0)
