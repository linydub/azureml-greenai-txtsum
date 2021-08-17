# (March) NLP Text Summarization Research
The research in this document is mainly on supervised Transformer-based (encoder-decoder) models for abstractive text summarization tasks. It also includes references on training and inference efficiency topics, model benchmarking, and other relevant findings.

**Text summarization tasks:**
* Language (e.g. English, multilingual)
* Domain-specific (e.g. financial news articles, medical research papers)
* Summarization context/task-specific/subtask (e.g. multi-document, long sequence, query-based, sentence compression)

Abstractive text summarization surveys (references):  
[Deep Learning Based Abstractive Text Summarization: Approaches, Datasets, Evaluation Measures, and Challenges](https://www.hindawi.com/journals/mpe/2020/9365340/)  
[Neural Abstractive Text Summarization with Sequence-to-Sequence Models](https://arxiv.org/abs/1812.02303)  
[A Survey of the State-of-the-Art Models in Neural Abstractive Text Summarization](https://ieeexplore.ieee.org/document/9328413)  


## Evaluation Metrics

The main qualities of an output summary to assess are the coherence/fluency of the generated output text, and how it reflects the original input text (accuracy, coverage/relevance).

**Current automated evaluation metrics limitations:** ROUGE
* Assess content selection mostly by lexical overlap. Problematic since abstractive summaries could express the same content with no lexical overlap.
* Designed to be used with multiple reference summaries per input, but most of the recent datasets only provide one reference.
* Weak correlation with human judgement while also failing to evaluate critical features like factual correctness/faithfulness.

**Non-automated evaluation:** [[2009.01325] Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
* Human preference vs reference summaries (comparison of human & model generated summaries)
* Axes of quality rating: Accuracy, coverage, coherence, overall

**Notes (supervised learning models):**

* Performance highly dependent on the quality of training dataset
* High cost of summarization dataset production
    * Not many public high quality datasets (domain/task-specific)
    * Most require human generated reference summaries
* Typically require manual human evaluation to validate model performance


## Dataset Candidates

Dataset quality: [Intrinsic Evaluation of Summarization Datasets](https://www.aclweb.org/anthology/2020.emnlp-main.649.pdf)

* **Multi-document:** Wikipedia Current Events Portal  
    [https://aylien.com/blog/multi-document-summarisation-and-the-wcep-dataset](https://aylien.com/blog/multi-document-summarisation-and-the-wcep-dataset)
* **Social media domain:** Webis-TLDR-17 Corpus (Reddit)  
[https://zenodo.org/record/1168855](https://zenodo.org/record/1168855)
* **Long sequence:** BigPatent  
    [https://evasharma.github.io/bigpatent/](https://evasharma.github.io/bigpatent/)


## Model Comparison

**Goals:**

* Evaluate tradeoffs between cost, carbon emission, accuracy, dataset size, model size, training & inference time, etc. (e.g. number of training samples vs model accuracy)
* Evaluate optimization/training strategies with tradeoffs (e.g. model size reduction vs accuracy loss vs training time for distillation) if applicable.
    * Implement ONNX Runtime for each model if applicable.
* Suggest architectures to implement based on a given use case with priorities and constraints (e.g. accuracy higher priority than training time, budget range, deployable for edge inference on mobile devices).

**Notes:**

* Note/rate difficulties or limitations in implementation/accessibility
    * Documentations/resources
    * Development time/effort vs benefit tradeoff
* Dataset size vs cost (supervised):  
    There could be additional costs for reference summary production when dealing with specialized domains. Models that require training/fine-tuning with large custom labeled datasets to meet a desired outcome could potentially be cost/time prohibitive.


## Model Info (Template)

**[Model Pre-training]**

* Pre-trained/Fine-tuned/From-scratch
    * *Pre-train objective(s) (e.g. MLM)
    * *Pre-train task(s) (e.g. text generation, machine translation)
    * Text summarization task
        * Method (abstractive/extractive)
        * Learning strategy (e.g. reinforcement, unsupervised)
        * Task-specialized/Subtask(s) (e.g. long sequence)
    * *Fine-tuned checkpoint (dataset info)
        * Medium/Format (e.g. blog article, novel, social media)
        * Domain(s) (e.g. sports, finance, academia, gaming)
        * Sequence length (e.g. document sentence/word count)
        * Single/Multi-document

**[Model Complexity]**

* Architecture (e.g. Transformer/encoder-decoder, LSTM/RNN)
    * Size (parameters, layers, memory usage)
        * Training
        * Post-training/Deployed
* ~~Other features (e.g. non-sequential structure -> easier parallelization)~~

**[Model Accuracy/Precision]**

* Performance evalulation metrics (e.g. JS divergence -> topic similarity, ROUGE)

**[Model Speed]**

* Training time (e.g. runtime, sec per batch, training steps)
* Inference time (CPU/GPU latency, sec per sample)

**[Model Power/Compute Consumption]**

* Training & Inference (e.g. GFLOPS/FPO)

**[Other Criteria]**

* **Carbon emission**
* Data preprocessing (e.g. tokenization method)
* Hardware (Training & Inference)
    * Reference specifications (e.g. 8 x A100 GPU, VRAM)
    * Minimum specification requirements
* Implementation frameworks/libraries/environments (e.g. HuggingFace, PyTorch)
* Hyperparameters (e.g. learning rate, batch size, momentum)
    * Stopping criterion (early stopping)
* Optimization/training strategies

## Model Candidates

### Baseline Models

**Implementation document: [Pretrained Models (TXTSUM)](https://docs.google.com/document/d/1DpBgdQ15dtYmVflVYpo7FKhvhtRIQUDSn_esZZBbEuc/edit#)**

**Notes:**

* Unsupervised methods have an advantage in training costs as they do not require labeled training datasets (reference summaries), but they generally score lower accuracies on the current automated evaluation metrics.
* The current text summarization Transformers are complex, large, and resource-intensive, which often make deployment of these models for edge computing challenging.
* Compare performance of existing fine-tuned models as is with no further training/fine-tuning (e.g. ProphetNET fine-tuned on Gigaword). Use them as baselines.

**Baseline (low-complexity unsupervised non-NN extractive):** TextRank, LexRank

* <span style="text-decoration:underline;">Paper:</span>
    * TextRank: [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
    * LexRank: [LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html)
* <span style="text-decoration:underline;">Code:</span>
    * [gensim: summarization.summarizer – TextRank Summariser](https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html)
    * [lexrank · PyPI](https://pypi.org/project/lexrank/)
    * Extractive Library? (lexrank, lsa, etc.): [https://github.com/miso-belica/sumy](https://github.com/miso-belica/sumy)

HipoRANK? [[2005.00513] Discourse-Aware Unsupervised Summarization of Long Scientific Documents](https://arxiv.org/abs/2005.00513) (2021-01)


### Pretrained Models

Transformer-based models (HuggingFace library): [https://huggingface.co/transformers/model_summary.html](https://huggingface.co/transformers/model_summary.html)

**Encoder-decoder:** [PEGASUS, T5, or BART], [ProphetNET]  
**Sparse attention:** [BigBird], [Longformer/LED], [Reformer/RED]

**[[PEGASUS: A State-of-the-Art Model for Abstractive Text Summarization]](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html)**  

* Low-resource summarization task performance better than previous “SOTA” results on 6 datasets with only 1000 examples
* **Pros:** Specifically designed for abstractive text summarization task, great sample efficiency for achieving “SOTA” accuracy
* **Cons:** Complex architecture (low explanability), large model size (memory/computational load prohibitive), high inference latency, labeled training data required (supervised learning)
* **Problem:** In prior work on pre-trained transformer-based models (BERT, GPT-2, T5), the self-supervised objectives have been somewhat agnostic to the down-stream application in favor of generality.
* **Hypothesis:** The closer the pre-training self-supervised objective is to the final target task, the better the fine-tuning performance. In PEGASUS, the gap-sentence generation objective designed for Transformer encoder-decoder models could improve fine-tuning performance on abstractive summarization.
* <span style="text-decoration:underline;">Time/cost estimates:</span> [PEGASUS evaluation time & cost estimates (AzureML)](https://docs.google.com/spreadsheets/d/1OJGhG2sBqxg29nEuawFtecL7y-3XPAM27Wb9E7TnZEE/edit#gid=0)
* <span style="text-decoration:underline;">Training results:</span> [PEGASUS training details and summarization results](https://paperswithcode.com/paper/pegasus-pre-training-with-extracted-gap/review/)
* <span style="text-decoration:underline;">Papers:</span>
    * [[1912.08777] PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
    * Distill-PEGASUS: [[2010.13002] Pre-trained Summarization Distillation](https://arxiv.org/abs/2010.13002)
    * BigBird-PEGASUS: [[2007.14062] Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
* <span style="text-decoration:underline;">Code:</span>
    * [https://github.com/google-research/pegasus](https://github.com/google-research/pegasus)
    * [Pegasus — transformers 4.5.0.dev0 documentation](https://huggingface.co/transformers/model_doc/pegasus.html)
    * ONNX Runtime:
        * [convert HuggingFace's Seq2seq models to onnx](https://stackoverflow.com/questions/66109084/how-to-convert-huggingfaces-seq2seq-models-to-onnx-format/66117248#66117248)
        * [microsoft/onnxruntime-training-examples: Examples for using ONNX Runtime for model training.](https://github.com/microsoft/onnxruntime-training-examples)
* <span style="text-decoration:underline;">Approach?:</span>
    * Use a large model for training, stop training early, then compress (quantization/pruning).
    * Use a distilled checkpoint fine-tuned on a similar dataset, further fine-tune the model on the new dataset.


## **[Resources]**

[Comparing Transformers on Few-Shot and Zero-Shot Multi-document Abstractive Summarization](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7720861/?report=classic) (PEGASUS, BART, T5)  
* All 3 transformer-based pre-trained models produced summaries with little discernible difference in quality after 10 examples.
* Model comparison was only on accuracy.

[[2010.13002] Pre-trained Summarization Distillation](https://arxiv.org/abs/2010.13002) (PEGASUS, BART)  
[[2004.06190] A Divide-and-Conquer Approach to the Summarization of Long Documents](https://arxiv.org/abs/2004.06190)  
[[2007.14062] Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)  

Content Hallucination:
[[2005.00661] On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661)

**GPU/CPU (Compute & Memory):**

The GPU memory for deep learning tasks are dependent on many factors like the number of trainable parameters in the network, batch size, floating-point format (FP16/FP32), number of activations, etc.

* [Estimating GPU Memory Consumption of Deep Learning Models](https://www.microsoft.com/en-us/research/uploads/prod/2020/09/dnnmem.pdf)
    * This empirical study has found that many DL job failures are due to the exhaustion of GPU memory, which leads to significant waste of computing resources.
* [https://lambdalabs.com/blog/choosing-a-gpu-for-deep-learning/](https://lambdalabs.com/blog/choosing-a-gpu-for-deep-learning/)

**Model Scaling:**
* [[2001.08361] Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (OpenAI)
* [[2002.11794] Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://arxiv.org/abs/2002.11794)
    * Increasing the model size can improve the efficiency of training and inference.
    * Larger models compress better (quantization, pruning). Heavily compressed large models achieve higher accuracy than lightly compressed smaller models.

**Low-Resource:**
* Extract-then-abstract: [[2103.00751] Long Document Summarization in a Low Resource Setting using Pretrained Language Models](https://arxiv.org/abs/2103.00751)
* [[2102.09397] Meta-Transfer Learning for Low-Resource Abstractive Summarization](https://arxiv.org/abs/2102.09397)
* [Abstractive Document Summarization in High and Low Resource Settings](https://www.research-collection.ethz.ch/handle/20.500.11850/425533)

**Energy/Carbon Emission:**
* [[2104.10350] Carbon Emissions and Large Neural Network Training](https://arxiv.org/abs/2104.10350)
* [[2002.05651] Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning](https://arxiv.org/abs/2002.05651)
* [Towards Accurate and Reliable Energy Measurement of NLP Models](https://www.aclweb.org/anthology/2020.sustainlp-1.19.pdf)
* [Estimation of energy consumption in machine learning](https://www.sciencedirect.com/science/article/pii/S0743731518308773)

**Evaluation/Metrics:**
* [[2010.05139] CDEvalSumm: An Empirical Study of Cross-Dataset Evaluation for Neural Summarization Systems](https://arxiv.org/abs/2010.05139)
* [[2010.07100] Re-evaluating Evaluation in Text Summarization](https://arxiv.org/abs/2010.07100)
* [[2007.12626] SummEval: Re-evaluating Summarization Evaluation](https://arxiv.org/abs/2007.12626)
* [[1908.08960] Neural Text Summarization: A Critical Evaluation](https://arxiv.org/abs/1908.08960) (2019-08)
* [A Framework for Word Embedding Based Automatic Text Summarization and Evaluation](https://www.mdpi.com/2078-2489/11/2/78/htm)

**Benchmarks:**
* [https://paperswithcode.com/task/text-summarization](https://paperswithcode.com/task/text-summarization)
* Computational cost/training time benchmarks:
    * [Benchmarks — transformers 4.5.0.dev0 documentation](https://huggingface.co/transformers/master/benchmarks.html)
    * ~~[https://www.openml.org/search?type=measure](https://www.openml.org/search?type=measure)~~
    * ~~[https://dawn.cs.stanford.edu/benchmark/index.html](https://dawn.cs.stanford.edu/benchmark/index.html)~~
    ~~(Only image classification and question answering)~~
    * [https://sotabench.com/](https://sotabench.com/)

Other NLP models focused on efficiency:
* [Advancing NLP with Efficient Projection-Based Model Architectures](https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html)
* [A language learning system that pays attention more efficiently: Researchers' new hardware & software system streamlines state-of-the-art sentence analysis](https://www.sciencedaily.com/releases/2021/02/210210133407.htm)
    * NLP models require heavy compute power, in part to the high memory demands of the attention mechanism. The problem worsens as NLP models grow more complex, especially for <span style="text-decoration:underline;">long sentences</span>.

### **[Other Documents]**

[PEGASUS evaluation time & cost estimates on AzureML (March)](https://docs.google.com/spreadsheets/d/1OJGhG2sBqxg29nEuawFtecL7y-3XPAM27Wb9E7TnZEE/edit#gid=0)  
[Reference compilation & notes (March)](https://docs.google.com/document/d/1Nd2GFBLvnsBECjiDzPlxKVqFnFfdVbm4cXe5WKclMOA/edit)  
