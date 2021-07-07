# Experiments
WIP!

## Environment Info
- Platform
- Hardware
- 

{env_table}

## Datasets Info
Sample count and sequence length.

| Dataset | Train | Val | Test | Source (words/tokens) | Target (words/tokens) |
| ------- | ----- | --- | ---- | :---: | :---: |
| CNN-DM | 287K	| 13K | 11K | 781/906 | 56/63 |
| XSUM | 204K | 11K	| 11K | 431/488 | 23/27 |
| SAMSum | 14.7K | 818 | 819 | 94/147 | 20/26 |

Reference:
https://paperswithcode.com/paper/hierarchical-learning-for-generation-with/review/?hl=30985  

### SAMSum
- SAMSum dataset similar to XSUM.
- Used XSUM hyperparameters when fine-tuning.
- 

## Models Info
{model_tables}

## Benchmarks
- Scale: Bart-Large vs Bart-Base on SAMSum, XSUM

## Comparisons
**Core:**  
- ZeRO optimization: Reduces GPU memory footprint, increases speed
- ZeRO offload/infinity (CPU): Reduces GPU memory footprint, speed trade-off
- Population based tuning over grid search (Hyperparameter tuning): Less runtime per trial, increases accuracy
- Floating point format (FP16- automatic mixed precision): Reduces , increases throughput
- Sparse Attention Kernel: Reduces GPU memory footprint, higher GPU memory efficiency
- 

**Other:**
- Parameter freezing (Freeze embedding modules,  Freeze encoder): Reduces GPU memory footprint, increases throughput
- Sortish sampling: Reduces padding, GPU memory footprint, accuracy trade-off
- Data Truncation: Reduces GPU memory footprint, accuracy trade-off
- Early stopping: Reduces runtime, accuracy trade-off
- 

### Hyperparameters
- Batch size: Increase memory usage, increase speed
- DS bucket sizes: Increase memory usage, increase speed
- 

## Trade-offs
There are lots of combinations to experiment and tune with, x memory results in decreased accuracy but potentially worth the speed up. Other things could influence effectiveness or accuracy like the corpus and model scale.

Trade-offs and benefits:  
{table}  
Some experiment benchmark results:  
{table}  

## Benchmark Results
R-1/R-2/R-L  

Reference:  