# aml-hf-txtsum
### NLP Text Summarization
This project includes a training sample and guide for training Hugging Face models on AzureML (with AML 2.0 CLI) similar to Hugging Face's SageMaker guides. The project also includes a case study that evaluates benefits and trade-offs of training strategies that could potentially reduce fine-tuning costs for pretrained encoder-decoder Transformer-based models on supervised abstractive text summarization. The empirical benchmark results produced from the case study experiments (including resource/carbon metrics & fine-tuning costs) will be made available publicly for users to use as a reference for model selection/development or other research.  
<br/>


#### Samples (by June 28th?):
-Hugging Face AML samples (scripts & notebook)  
*Similar to the Hugging Face Amazon SageMaker guides  
https://huggingface.co/transformers/master/sagemaker.html  
https://huggingface.co/blog/sagemaker-distributed-training-seq2seq  
-Will link this repository & guides in trained models pages for exposure once basic sample is ready  
https://huggingface.co/henryu-lin/bart-large-samsum  
-Repo code & docs cleanup for public release  



#### Case study (by June 30th?):
-Retrieve & compile experiment benchmark results & comparisons (with resource/carbon metrics & cost)  
-Pretrained Seq2Seq model comparisons on different model scale, pretrained objectives, pretrained corpus  
-Core cost reduction experiments: baseline (PyTorch), sparse attention (LED, BigBird), parameter freezing (embeds, encoder), deepSpeed (ZeRO 2, ZeRO 3, CPU Offload)  



#### Extras:
-Summarization experiments' modified Hugging Face trainer documentation  
-Fine-tuning guide (tips & insights)  
-Additional cost efficient training/fine-tuning strategy experiments  
-Case study in-depth evaluation  
*If these experiments are worthwhile, I'll finish evaluations & do a proper paper/blog write-up..  
