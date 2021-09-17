# Fine-tuning Benchmarks [WIP]

## Base Models
### Environment Details
| key | value |
| --- | ----- |
| Region | US West 2 |
| AzureML Compute SKU | {compute_details["size"]} |
| GPU Device | 4 x NVIDIA V100 16GB |

### SAMSum
| Model | ROUGE (1/2/L) | Cost ($) | Run Duration | Memory Usage | Energy Usage | Emissions | Details |
| ----- | ------------- | -------- | ------------ | ------------ | ------------ | --------- | ------- |
| BART-Base | X/X/X | $X USD | Xh Xm Xs | X GB | X kJ | X | [YAML Specs](./temp) |

### XSum

### CNN/DailyMail

### Reddit-TIFU

## Large Models
### Environment Details
| key | value |
| --- | ----- |
| Region | US West 2 |
| AzureML Compute SKU | {compute_details["size"]} |
| GPU Device | 8 x NVIDIA V100 32GB (NVLink) |

### SAMSum
| Model | ROUGE (1/2/L) | Cost ($) | Run Duration | Memory Usage | Energy Usage | Emissions | Details |
| ----- | ------------- | -------- | ------------ | ------------ | ------------ | --------- | ------- |

## DeepSpeed
### Environment Details
| key | value |
| --- | ----- |
| Region | US West 2 |
| AzureML Compute SKU | {compute_details["size"]} |
| GPU Device | 4 x NVIDIA V100 16GB |

### Experiment Details
| key | value |
| --- | ----- |
| Model | BART-Large |
| Dataset | SAMSum |
| Hyperparameters | [YAML Specs](.) |
| DeepSpeed Config | [JSON](.) |

### Results
| key | PyTorch | DeepSpeed ZeRO 2 |
| --- | ------- | ---------------------- |
| ROUGE (1/2/L) | X/X/X | X/X/X |
| Cost ($) | $X USD | $X USD |
| Run Duration | Xm Xs | Xm Xs |
| Memory Usage | X GB | X GB |
| Energy Usage | X kJ | X kJ |
| Emissions | X | X |
