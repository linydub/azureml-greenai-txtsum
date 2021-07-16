# Case Study: Evaluation of Cost Effective Strategies for Pretrained SOTA NLP Transformers on Supervised Abstractive Text Summarization (June)

WIP

### **Setup:**

We will be running our training experiments using the AzureML platform.  
Compute cost on AzureML is calculated by the SKU hourly price and training duration.  

AzureML SKU: ND40rs_v2 (NVIDIA V100 32GB x 8)  
Reference model: Bart-Large  
-406M parameters, 24-layer, 1024-hidden, 16-heads

Reference dataset: SAMSum  
-Samples: 14.7k training, 818 validation, 819 test  
-Sequence length (BPE tokens): 147 Input/Source, 26 Output/Target  

**Efficiency strategies:**



* Floating point format: The use of FP16 (mixed precision) has been well-established and provides significant training speed increase (~2x) over FP32 with more recent GPU hardware (Nvidia V100, A100). A major thing to consider is the floating point format that the model was pretrained with, as you will need FP16 weights of the model to fine-tune it on FP16 ([Compiling data on how models were pre-trained: fp16, fp32, bf16](https://discuss.huggingface.co/t/compiling-data-on-how-models-were-pre-trained-fp16-fp32-bf16/5671)). Nvidia’s A100 supports training in a new TensorFloat-32 mode (TF32).

    Experiment: We will be using [Hugging Face](https://huggingface.co/transformers/master/main_classes/trainer.html) library’s built in trainer to train on FP32 and FP16 (AMP).


    Result summary: 

* Hyperparameter tuning: Hyperparameter search with early termination and warm starting can reduce unnecessary computations. Popular methods are grid/random search and bayesian optimization. Another method is [population based training (PBT)](https://arxiv.org/abs/1711.09846), a type of heuristic search. This method could potentially save unnecessary computations by continuing to tune from previous successful model checkpoints instead of restarting from scratch every iteration.

    Experiment: We will be using AzureML’s [HyperDrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) for random search and [RayTune](https://docs.ray.io/en/master/tune/index.html) library for PBT.


    Result summary: 

* Early stopping: Optimize for loss or accuracy/performance, stop training when metric stops improving or worsens. Could save costs by stopping when metric barely improves (0.1~0.5 early stopping patience) or hits an acceptable metric score range. Note that the compute cost increases alongside the frequency of model evaluation.

    Experiment: Modify Hugging Face’s seq2seq trainer, add an [early stopping callback](https://huggingface.co/transformers/master/main_classes/callback.html), optimize on ROUGE-1 metric.


    Result summary: 

* DeepSpeed: Model parallelism and distributed data parallelism using zero optimization and zero-offloading/zero-infinity. Bucket size should be maximized after maximizing batch size to increase GPU usage efficiency and training speed. Train larger models (11 billion param T5) using cpu or nvme offloading (zero3).

    Experiment: [DeepSpeed](https://www.deepspeed.ai/features/) integration with Hugging Face library.


    Result summary: 

* Sortish sampling: Should reduce the amount of padding, increasing speed. Should use full sorting during evaluation since not optimizing params, so shouldn’t affect model performance with an increase in evaluation throughput speed.

    Experiment: 


    Result summary: There was no notable difference in training speed nor inference speed. Validation and testing dataset size was relatively small. Need further testing on a bigger dataset and with longer sequence length & different sequence distribution.

* Parameter freezing: Freezing parameters reduces the amount of param optimization done, or the number of trainable parameters. This fine-tuning strategy depends on the model’s pretrained and fine-tuning corpus/dataset (model vocab, domain, dataset similarity, etc) for quality of generated summaries.

    Experiment: Modified Hugging Face trainer, added param freezing for embedding and encoder modules (torch.nn.Module).


    Result summary: Significant decrease in GPU memory footprint (~5 GBs with encoder freezing). Surprisingly, higher accuracy within the first epoch when freezing embeddings module (ROUGE-1 55.162 vs 53.058). Further optimization by increasing DeepSpeed bucket size from the extra memory should speed up training even more.

* Sparse Attention: For longer sequences, memory efficiency.

    Experiment: LED vs Bart, Pegasus vs BigBird-Pegasus


    Result summary: 


**Training Tips:**



* Maximize resource utilization: Increase resource usage efficiency (GPU, CPU compute and GPU memory). Main thing to consider optimizing is batch size (larger per device batch size results in more memory usage). Faster training usually results in cheaper training costs since most cloud data centers charge VMs on a timed basis, not power usage.
* Compute resources (AzureML): Using low priority compute clusters instead of dedicated clusters during the model development process could save a significant amount of computing costs (one V100x8 node: $22.03 vs $4.41 per hour). When using low priority custers, consider doing model checkpoints in case the cluster gets preempted during training. Check [Pricing - Batch for Linux VMs](https://azure.microsoft.com/en-us/pricing/details/batch/) for updated price comparisons.
* Memory: Common constraint is GPU memory (VRAM) which directly limits the batch size and model size (number of trainable params). OOM failures cost a significant amount of waste in computing, so make sure to leave some memory for fluctuations. A Popular strategy to increase the effective batch size with limited GPU memory is gradient accumulation (speed trade-off), and if using multiple GPUs, distributed data parallelism and model parallelism. A more recent solution is DeepSpeed zero optimization and cpu offloading.

    Autoscale batch size: [https://github.com/BlackHC/toma](https://github.com/BlackHC/toma)

* *Knowledge/expertise: Having a better understanding of model’s architecture/complexity saves time & compute costs, especially during the hyperparam tuning process, less blind tuning hyperparams, smaller search space.

**Development process (fine-tuning):**

-Data preparation is the most important and time consuming part. Common summarization problems include content hallucination (lots of intrinsic hallucinations from models fine-tuned on XSUM compared to CNN/DailyMail, could be result of data preprocessing, possibly from sequence truncation on source document during tokenization process).

-Pretrained model selection based on task/constraints. Testing/evaluation of initial pretrained models performances. Also evaluate deployment/inference (resource usage, latency, and costs) before inference optimizations.

-Hyperparameter tuning/search is often used and consumes a significant portion of training compute costs. Usually maximize batch size (and bucket size if using ZeRO optimization) then tune learning rate.  
*Tools such as AutoScale batch size and DeepSpeed memory estimator could help.

([[1804.00247] Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)) Main hyperparameters to optimize are batch size and learning rate. Bigger batch size leads to better accuracy and faster training time, so the batch size should be maximized while leaving some memory to prevent hitting OOM. Batch size should be a power of 2 ([How big should my language model be ?](https://huggingface.co/calculator/)). 

-Evaluation/comparison of results for model selection.

**[References]**

---

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

## Carbon Footprint vs Monetary Cost
- Minimizing monetary cost does not always lead to a reduction in carbon emissions
- Usually when minimizing cost on cloud computes, one would maximize the resource usage of the compute VMs

### Hyperparameter Tuning
Hyperparameter tuning method, some common hyperparams to tune are learning rate, weight decay, label smoothing. Decoder params includes number of beams, length penalty, max and min generation length.
Schedulers and optimizers, such as adafactor could reduce memory usage which enables training larger models or batch sizes.

### Truncation
type of tokenization method, sequence length distribution of source and target, this will affect the accuracy