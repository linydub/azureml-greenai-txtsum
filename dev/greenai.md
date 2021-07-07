## GreenAI Initiative
WIP!

## Model Selection
Different pretraining objectives, pretraining corpus. Some models found on huggingface are unable to train in mixed precision.  
https://discuss.huggingface.co/t/compiling-data-on-how-models-were-pre-trained-fp16-fp32-bf16/5671

## Limitations of Current Models
Content hallucination, factual incorrectness  
Good examples:  
Minor adjustments results in poor summarization:  
Other bad examples:  

## Carbon Footprint vs Monetary Cost
- Minimizing monetary cost does not always lead to a reduction in carbon emissions. Other factors mentioned like location.
- Usually when minimizing cost on cloud computes, one would maximize the resource usage of the compute VMs.

### Maximizing Memory Usage
*Usually maximize batch size (and bucket size if using ZeRO optimizations) then tune learning rate.
Tools such as AutoScale batch size and DeepSpeed memory estimator could help.

### Hyperparameter Tuning
Hyperparameter tuning methods influence the cost, some common hyperparams to tune are learning rate, weight decay, label smoothing. Decoding includes number of beams, length penalty, max and min generation length.
Schedulers and optimizers, such as adafactor could reduce memory usage which enables training larger models or batch sizes.

### Truncation
When truncating, you should be aware of x tokenization and check sequence length distribution of source and target tokens. This will affect the accuracy.

## Knowledge/Skill


## Future Research

