# draft v1

#transformers 4.6.1
#pip freeze | cut -d'=' -f1 | xargs -n1 pip install -U
#nvidia-smi
#torch.cuda.is_available()
#torch.cuda.empty_cache()
import nltk
nltk.download('punkt')
import numpy as np
#import pandas as pd
import torch
import argparse
import mlflow
#import azureml.core
#import logging
#import sys
#import time
import os
from datasets import load_from_disk, load_metric
from transformers.integrations import MLflowCallback#, AzureMLCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    #set_seed,
    #AutoConfig,
    #HfArgumentParser
)

# td: param config
class MLflowNoConfigCallback(MLflowCallback):

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.

                This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy
                whatever is in :class:`~transformers.TrainingArguments`'s ``output_dir`` to the local or remote
                artifact storage. Using it without a remote storage will just copy the files to your artifact location.
        """
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1"}:
            self._log_artifacts = True
        if state.is_world_process_zero:
            self._ml_flow.start_run()
            combined_dict = args.to_dict()
            # remove model params (aml limit 100 params)
            """
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            """
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{value}" for key "{name}" as a parameter. '
                        f"MLflow's log_param() only accepts values no longer than "
                        f"250 characters so we dropped this attribute."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))
        self._initialized = True

# HfArgumentParser wip
parser = argparse.ArgumentParser(description="HuggingFace Trainer (WIP)")
parser.add_argument(
    "--data",
    type=str,
    default=None,
    help="data path")
# cmd args
args = parser.parse_args()

# download model (cache)
model_name = "google/pegasus-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir = "./cache")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = "./cache")

# td? FileSystems integration for cloud storage (abfs for Azure Blob service)
# load dataset from workspace
data = load_from_disk(args.data) #keep_in_memory=True <- mount

# train/eval/test sets
train_data = data["train"].select(range(256))
eval_data = data["validation"].select(range(64))
test_data = data["test"].select(range(64))

# max sequence length
# (xsum: 512, 56), (reddit-tifu: 512, 128)
max_source_length = 512
max_target_length = 56

# td parse: source & target columns
def preprocess_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_eval = eval_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

len(tokenized_train)

# task performance metric
metric = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # rouge sentence newline formatting (data post processing)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # mean summary length metric
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["sum_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# For optimization, both pre-training and fine-tuning used Adafactor (Shazeer & Stern, 2018) with square root learning rate decay and dropout rate of 0.1.
#num_beams = 8, # PEGASUS -> 8
#gradient_accumulation_steps = (256/batch_size/nodes) # PEGASUS effective batch size -> 256

# early stopping
#load_best_model_at_end = True,
#metric_for_best_model="eval_rouge2",
#greater_is_better=True,
#early_stopping = True,
#val_check_interval = 0.25, # <- evaluation_strategy?

#freeze_embeds = True,
batch_size = 4
train_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    gradient_accumulation_steps = 256 / batch_size,
    adafactor = True, # PEGASUS -> Adafactor
    fp16 = False, # hf-PEGASUS ONLY FP32
    label_smoothing_factor = 0.1,
    weight_decay = 0.01,
    learning_rate = 0.0001, # (1e-4) -> 256 effective batch size
    num_train_epochs = 2.0,
    evaluation_strategy = "epoch",
    logging_strategy = "steps",
    logging_steps = 1,
    logging_dir='./logs',
    save_strategy = "epoch",
    save_total_limit = 1,
    output_dir = "./outputs/pegasus-dev",
    overwrite_output_dir = True,
    predict_with_generate = True,
    #do_eval = True, # changed to eval before fine-tuning
    do_train = True,
    do_predict = True
)

#optimizers=[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]
#[torch.optim.Adam(params=model.parameters(), lr=args.learning_rate), None]
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    #optimizers = optimizers,
    args = train_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_eval,
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

# azureml-mlflow param limit = 100
trainer.remove_callback(MLflowCallback)
trainer.add_callback(MLflowNoConfigCallback)
#trainer.remove_callback(AzureMLCallback)

# add EarlyStoppingCallback

# evaluate before fine-tuning
if train_args.do_eval:

    metrics = trainer.evaluate(
        max_length = max_target_length, #data_args.val_max_target_length
        metric_key_prefix = "eval",
        num_beams = 8 #data_args.num_beams
    )
    metrics["eval_samples"] = len(tokenized_eval)
    #max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

# train 
if train_args.do_train:

    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_train)
    #max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    #metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# inference summary generation
if train_args.do_predict:

    results = trainer.predict(
        tokenized_test,
        metric_key_prefix = "test",
        max_length = max_target_length, #data_args.val_max_target_length
        num_beams = 8 #data_args.num_beams
    )
    metrics = results.metrics
    metrics["test_samples"] = len(tokenized_test)
    #max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(tokenized_test)
    #metrics["predict_samples"] = min(max_predict_samples, len(tokenized_test))

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    # output text summaries file
    if trainer.is_world_process_zero():
        if train_args.predict_with_generate:
            summary_texts = tokenizer.batch_decode(
                results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True # docs
            )
            summary_texts = [text.strip() for text in summary_texts] # necessary.?
            summary_file = os.path.join(train_args.output_dir, "generated_summaries.txt")
            with open(summary_file, "w") as writer:
                writer.write("\n".join(summary_texts))