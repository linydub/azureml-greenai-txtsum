# (June) Case Study: Evaluation of Training Strategies for Pretrained NLP Transformers on Supervised Abstractive Text Summarization
Draft

## Setup
The fine-tuning experiment examples were ran on the AzureML platform. Compute costs on AzureML were calculated by the SKU hourly price and run duration.  

AzureML Compute SKU:
- ND40rs_v2 (8 x V100 32GB)
draft-docs
Reference model:
- BART-Large (encoder-decoder)  
    - 406M parameters, 24-layer, 1024-hidden, 16-heads

Reference dataset:
- SAMSum (dialogue summarization)  
    - Samples: 14.7k training, 818 validation, 819 test  
    - Sequence length (BPE tokens): 147 source, 26 target  

## Experiments
* Floating point format: FP16 could provide training speed increase (~2x) over FP32 on "recent" GPU hardware (Nvidia V100, A100). [How the models were pre-trained](https://discuss.huggingface.co/t/compiling-data-on-how-models-were-pre-trained-fp16-fp32-bf16/5671) should be considered since FP16 weights of the model would be needed to fine-tune it in FP16. Nvidia’s A100 supports training in a new TensorFloat-32 mode (TF32).

    Method: Use [HuggingFace's trainer](https://huggingface.co/transformers/master/main_classes/trainer.html) to fine-tune in FP32 and FP16 (AMP).

    Result summary: 

* Hyperparameter tuning: Hyperparameter search with early termination and warm starting can reduce unnecessary computations. Popular methods are grid/random search and bayesian optimization. Another method is [population based training (PBT)](https://arxiv.org/abs/1711.09846), a type of heuristic search. This method could potentially save unnecessary computations by continuing to tune from previous successful model checkpoints instead of restarting from scratch every iteration.

    Method: Used AzureML’s [HyperDrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) for random search and [RayTune](https://docs.ray.io/en/master/tune/index.html) library for PBT.

    Result summary: 

* Early stopping: Optimize on loss or accuracy/performance, stop training when metric stops improving or worsens. Could save costs by stopping when metric barely improves (0.05~0.1 early_stopping_threshold) or hits within an acceptable performance range. Compute cost is affected by frequency of evaluation.

    Method: Use [early stopping callback](https://huggingface.co/transformers/master/main_classes/callback.html), optimize on loss or ROUGE metric.

    Result summary: 

* DeepSpeed: Zero optimizations and zero-infinity/offload.

    Method: Use Hugging Face's [DeepSpeed](https://www.deepspeed.ai/features/) integration. Compare zero optimizations (stage 0/1/2/3).

    Result summary: 

* Sortish sampling: Should reduce the amount of padding, increasing speed. Should use full sorting during evaluation since not optimizing params, so shouldn’t affect model performance with an increase in evaluation throughput speed.

    Method: 

    Result notes: There was no notable difference in training speed nor inference speed. Validation and testing dataset size was relatively small. Need further testing on a bigger dataset and with longer sequence length & different sequence distribution.

* Parameter freezing: Freezing parameters reduces the amount of param optimization done. Depends on model’s pretrained corpus and fine-tuning corpus (model vocab, domain, dataset similarity, etc).

    Method: 

    Result summary: Significant decrease in GPU memory footprint (reduced ~5 GBs with encoder freezing). On average, higher ROUGE scores at each epoch when only freezing embeddings module (3rd epoch: ROUGE-1 55.162 vs 53.058).

* Sparse Attention: For longer sequences, memory efficiency.

    Method: LED vs Bart, Pegasus vs BigBird-Pegasus, also compare on datasets with longer sequence lengths

    Result summary: 

* Data truncation: Type of tokenization method, sequence length distribution of source and target will affect accuracy.  

    Method: 

    Result summary: 

## Training Tips
* Maximize resource utilization: Increasing resource usage efficiency (GPU, CPU compute and GPU memory) should lead to less training time. Faster training reduces training costs since most cloud data centers charge VMs on a timed basis, not compute/power usage.

* Compute resources (AzureML): Using low priority compute clusters instead of dedicated clusters during development could save computing costs (8 x V100 32GB node: $22.03 vs $4.41 per hour). When using low priority custers, consider doing model checkpoints in case that the cluster gets preempted during training. Check [Pricing - Azure Machine Learning](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) for updated price comparisons.

* Memory: A common constraint is GPU memory (VRAM) which limits the batch size and model size (number of trainable params). OOM failures results in a significant amount of waste of computing, should leave some extra memory. Could increase the effective batch size by using gradient accumulation (trade-off training speed for extra memory), and if using multiple GPUs, distributed data parallelism and model parallelism. A more recent solution is DeepSpeed's zero optimizations and cpu/nvme offloading.

* Knowledge/expertise: Deep understanding saves time & compute costs during development, e.g. in hyperparam tuning process (less blind tuning, smaller search space).

* 

### Development process (pretrained models)
* Data preparation is usally the most important and time consuming part. Problems of text summarization include content hallucination. Significantly more intrinsic hallucinations in summaries generated from models fine-tuned on XSum compared to CNN/DailyMail, could be result of dataset's source & target qualities, could also be influenced by data preprocessing.  

* Select pretrained models to fine-tune for scenario/tasks. Evaluate/compare model performance/efficiency/costs before fine-tuning, could also test/evaluate models' potential performance after inference optimizations.

* Select hyperparameter tuning method. Hyperparameter tuning/search consumes a significant portion of training compute costs. Some common hyperparams to tune are batch size, learning rate, number of epochs, weight decay, label smoothing. Common decoder params are number of beams, length penalty, max and min generation length. Different schedulers and optimizers could have different interactions with hyperparameters. Usually maximize batch size (and optimize all_gather & reduce bucket size if using ZeRO optimizations) then tune learning rate. Tools that [autoscale batch size](https://github.com/BlackHC/toma) or estimate memory usage could help.  
Bigger batch size leads to [better accuracy and faster training time](https://arxiv.org/abs/1804.00247), so the batch size should be maximized. Batch size should generally be a power of 2.

* Perform model inference optimizations for the specified deployment method (edge/cloud, real-time/batch), platforms/hardwares.

* Evaluate the optimized fine-tuned models (final inference performance, resource usage, estimated costs), could also evaluate the model's retraining cost/time/difficulty.

## References
*  
