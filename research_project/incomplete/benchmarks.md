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
