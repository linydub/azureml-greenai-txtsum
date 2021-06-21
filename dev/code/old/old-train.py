import nltk
nltk.download('punkt')
import numpy as np
import torch
import argparse
#import mlflow
from datasets import load_from_disk, load_metric
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
#from transformers.integrations import MLflowCallback, AzureMLCallback
# by default Trainer will use callbacks if installed

# td: hfargumentparser
parser = argparse.ArgumentParser(description="HuggingFace Trainer (WIP)")
parser.add_argument(
    "--data",
    type=str,
    default=None,
    help="data path")
parser.add_argument(
    "--model",
    type=str,
    default="google/pegasus-xsum",
    help="model path")

# cmd args
args = parser.parse_args()

# load dataset from workspace
data = load_from_disk(args.data)

#data.train_test_split(test_size=0.1)
train_data = data.select(range(128))
val_data = data.select(range(1000, 1064))
#test_data = data.select(range(2000, 2064))

# download model (cache)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# max sequence length
# (xsum: 512, 56), (reddit-tifu: 512, 128)
max_source_length = 512
max_target_length = 56
def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_val = val_data.map(preprocess_function, batched=True)
#tokenized_test = test_data.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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

train_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size = args.batch_size,
    #gradient_accumulation_steps = 256/batch_size, # effective batch size = 256 (256/batch_size/nodes)
    adafactor = True,
    fp16 = False,
    label_smoothing_factor = 0.1,
    weight_decay = 0.01,
    learning_rate = args.learning_rate,
    num_train_epochs = args.epochs,
    evaluation_strategy = "epoch",
    logging_strategy = "steps",
    logging_steps = 8,
    logging_dir='./logs',
    save_strategy = "epoch",
    save_total_limit = 1,
    output_dir = "./outputs/hf-trial",
    overwrite_output_dir = True,
    predict_with_generate = True,
)

# azureml-flow version incompatibility: transformers v4.6.0.dev
# azureml-mlflow: https://github.com/huggingface/transformers/pull/10071 mlflow param overflow
# mlflow.exceptions.RestException: BAD_REQUEST: Response: {'Error': {'Code': 'UserError', 'Severity': None, 'Message': 'A field of the entity is over the size limit. FieldName=Parameters, Limit=100, Size=148. See https://aka.ms/azure-machine-learning-limits for service limits documentation.', 'MessageFormat': None, 'MessageParameters': None, 'ReferenceCode': None, 'DetailsUri': None, 'Target': None, 'Details': [], 'InnerError': None, 'DebugInfo': None}, 'Correlation': {'operation': '97dc438dde54304987b7615e3ff523fd', 'request': '17d9cfdbe83ad14a'}, 'Environment': 'eastus', 'Location': 'eastus', 'Time': '2021-04-30T16:16:38.6563505+00:00', 'ComponentName': 'mlflow', 'error_code': 'BAD_REQUEST'}

# td: onnxruntime
trainer = Seq2SeqTrainer(
    model,
    #optimizers = optimizers,
    train_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
    #callbacks: []
)
# evaluate before fine-tuning
trainer.evaluate()

# fine-tune
trainer.train()

# td generate comparison table (score.py)
#trainer.predict(tokenized_test)