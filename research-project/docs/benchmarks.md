# Benchmarking and Carbon Accounting in AzureML
Draft

## Task Overview
`{task_description}`

### Dataset details
Dataset sample count and sequence length (BPE tokens):

| Dataset | Samples (train/val/test) | Source (words/tokens) | Target (words/tokens) |
| ------- | ------------------------ | --------------------- | --------------------- |
| CNN-DM  | 287K/13K/11K | 781/906 | 56/63 |
| XSum    | 204K/11K/11K | 431/488 | 23/27 |
| SAMSum  | 14.7K/818/819 | 94/147 | 20/26 |

Reference:
https://paperswithcode.com/paper/hierarchical-learning-for-generation-with/review/?hl=30985  

### Environment details
`{requirements}`

| OS | GPU | CPU |
| -- | --- | --- |

### Model details
| Model | Params | Corpus |
| ----- | ------ | ------ |

### Hyperparameter configs
`{hyperparams.json}`

## Experiments
**Core:**  
- ZeRO optimizations (stage 1/2/3): Reduces GPU memory usage, *increases speed
- ZeRO offload/infinity (CPU): Reduces GPU memory usage, reduces speed
- Hyperparameter tuning (population based tuning vs grid search): Less total runtime & compute, *better accuracy
- Floating point format (FP32 vs FP16): Reduces GPU memory usage, increases speed
- Sparse Attention: Reduces GPU memory usage, *affects accuracy

**Other:**
- Parameter freezing (embedding, encoder modules): Reduces GPU memory footprint, increases throughput, *affects accuracy
- Sortish sampling: Reduces GPU memory usage, *reduces accuracy
- Data Truncation (sequence length): Reduces GPU memory usage, reduces accuracy
- Early stopping: Reduces runtime, *reduces accuracy
- 

Generally applicable in most cases, some are conditional and have trade-offs, *asterisks prefix = situational, accuracy refers to task performance  

## CNN/DailyMail, XSum, SAMSum Benchmarks
| Model | Performance | Speed | Resource Usage | Carbon Emission | Cost ($) |
| ----- | ----------- | ----- | -------------- | --------------- | -------- |

---


# GreenAI Initiative
## NLP Text Summarization
### Limitations of Current Pretrained Models and Performance Evaluation Metrics
- **Content hallucination & factual incorrectness:**

    Good summary examples:  
    ```
    {bart-large-cnn: document, summary}
    ```
    Minor adjustments in input document results in very different/poor generated summary:  
    ```
    'words replaced'
    {before, after}
    ```
    Other bad summary examples:  
    ```
    {pegasus-xsum-finance: crypto, environment, carbon, ethereum}
    ```

-  

## Cost-Benefit Analysis of NLP MLOps on Cloud Platforms
<p align="center">
  <img src="../images/GreenAI.png" alt="cba diagram"/>
</p>

### Monetary Cost vs. Environmental Cost
- Minimizing monetary cost does not always lead to a reduction in carbon emissions
- Usually when minimizing training cost on cloud computes, one could reduce the training duration by maximizing the resource usage of the compute target
- 

### Cloud Service Provider & Consumer Influence on Carbon Emissions
-  


---


# (May) Case Study: Carbon Footprint of Current SOTA Pretrained NLP Transformers on Supervised Abstractive Text Summarization
Goals from the initial GreenAI document:  
-   Develop a methodology for operational lifecycle analysis across a ML project that maps key efficiency/power metrics, cost ($), carbon, and accuracy.
-   Compare the cost/benefit for different baseline architectures.
-   Create public-facing samples to provide an easy starting point for customer POC, which can then be customized for specific scenarios.

## Scope
- NLP Task: Abstractive text summarization (English)
- Architecture: Transformers (encoder-decoder)
- Training strategy: Supervised, fine-tune pretrained models (transfer learning)

*Prefix (asterisk) are additional items outside of core scope.

## Metrics
Reference platform/service: Azure (cloud/datacenter)

Cost (monetary):
-   Compute (dollars/hour): VM charged on a timed basis independent of resource usage
-   Other services (storage, *data prep)

Quality (task performance):
-   Standard (accuracy & precision): ROUGE
-   *SummEval ([[2007.12626] SummEval: Re-evaluating Summarization Evaluation](https://arxiv.org/abs/2007.12626))\
    Repo: <https://github.com/Yale-LILY/SummEval#evaluation-toolkit>

Speed:
-   Runtime (seconds)
-   Throughput (samples/second, tokens/second)

Memory:
-   RAM, VRAM (GBs)
-   Disk space (GBs)

Compute:
-   GPU, CPU (utilization %)
-   FPOs
-   Energy (joules)
-   Carbon emission (CO₂eq - lbs)

## Samples
- End-to-end notebook: Sample guide using efficient strategies validated by experiment results for a scenario
- Scope: Training, *inference (deployment, maintenance)
- *Cloud endpoint example: Realtime logging of model's inference carbon emission

### Setup:
- AzureML 2.0 CLI
- Hugging Face (PyTorch)
- MLflow
- DeepSpeed
- ONNX Runtime (Training, *Inference, *Quantization)
- NVIDIA Triton
- *[Experiment-Impact-Tracker](https://github.com/Breakend/experiment-impact-tracker)/[CodeCarbon](https://github.com/mlco2/codecarbon)

## Replications
Models are fine-tuned with hyperparameter values mentioned in their paper or otherwise with sensible default values.  

- Models: Pegasus, Bart, T5, ProphetNET, *[UniLM](https://github.com/microsoft/unilm)  
*Sparsity:  BigBird-Pegasus, LED
- Datasets: XSUM, CNN/DailyMail, Reddit-TIFU, SAMSum, *Reddit (subreddit: finance, gaming)  
*Long sequence: BigPatent, PubMed

## Experiments
- Compare models of comparable size on public datasets
- *Evaluate different pretraining corpus and pretraining methods/objectives (transfer learning)
- *Compare environments/hardwares/platforms
- *Compare tools/frameworks/libraries/methods
- Evaluate model cost/power/memory/time/sample efficiency (benchmarks)
    - *Low-resource cases
    - *Specific cases (multi-document, long sequences, domain-specific, etc.)

### Parameters
- Model: architecture, size (params/layers/heads), checkpoint
    - Data (pretrained & training corpus): quantity, qualities
- Hardware: datacenter/*consumer, GPU/*CPU/*TPU, single/multi-gpu, single/multi-node
    - GPU (Nvidia T4, P100, V100, A100, *RTX 2000/3000s)
- Tools/Frameworks/Libraries: PyTorch, DeepSpeed, ONNX Runtime, *Horovod, *FairScale, *TorchScript, *TensorFlow
- Training Methods: parameter/layer freezing, early stopping, *distillation, *pruning, *ensemble, *quantization aware training
    - Floating-point format: FP32, FP16 (AMP), *TF32
    - *Optimizer: AdamW, Adafactor
    - *Hyperparameter tuning: Bayesian optimization, grid/random search, PBT, etc.

### Findings
- Training/development tips and insights
- Problems and/or difficulties during development/iterative process
- Empirical results (tables/graphs) & source code (github repo)
- Trade-off evaluations

## Resources
**Literature review (March/April):**  

[Text Summarization Research](https://docs.google.com/document/d/1NYVLTFWWFR3RrBWE7PD8VyLcs4S7CHncFivojKF7A60/edit)  
[Pretrained Models (TXTSUM)](https://docs.google.com/document/d/1DpBgdQ15dtYmVflVYpo7FKhvhtRIQUDSn_esZZBbEuc/edit#)  
[PEGASUS evaluation time & cost estimates (AzureML)](https://docs.google.com/spreadsheets/d/1OJGhG2sBqxg29nEuawFtecL7y-3XPAM27Wb9E7TnZEE/edit#gid=0)
