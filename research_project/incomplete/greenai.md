## GreenAI Initiative
WIP

## Model Selection
Different pretraining objectives, pretraining corpus. Some models on huggingface are unable to train in FP16.  
https://discuss.huggingface.co/t/compiling-data-on-how-models-were-pre-trained-fp16-fp32-bf16/5671

## Limitations of Current Models
Content hallucination, factual incorrectness  
Good examples:  
Minor adjustments results in poor summarization:  
Other bad examples:  

## Increasing Efficiency
- 
- 

## Carbon Footprint vs Monetary Cost
- Minimizing monetary cost does not always lead to a reduction in carbon emissions
- Usually when minimizing cost on cloud computes, one would maximize the resource usage of the compute VMs

### Maximizing Memory Usage
*Usually maximize batch size (and bucket size if using ZeRO optimizations) then tune learning rate.
Tools such as AutoScale batch size and DeepSpeed memory estimator could help.

### Hyperparameter Tuning
Hyperparameter tuning methods influence the cost, some common hyperparams to tune are learning rate, weight decay, label smoothing. Decoding includes number of beams, length penalty, max and min generation length.
Schedulers and optimizers, such as adafactor could reduce memory usage which enables training larger models or batch sizes.

### Truncation
When truncating, you should be aware of x tokenization and check sequence length distribution of source and target tokens. This will affect the accuracy.

## Knowledge/Skills


## Future Research

More experiments on power efficient training/fine-tuning


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
