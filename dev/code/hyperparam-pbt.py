# https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# first on partial dataset
train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10)
# then full dataset
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

best_run
# results
BestRun(
    run_id='2',
    objective=0.5215162259225145,
    hyperparameters={
        'learning_rate': 4.357724525964853e-05,
        'num_train_epochs': 2,
        'seed': 38,
        'per_device_train_batch_size': 32
        }
)

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

# params
hyperparameters={
    'learning_rate': 4.357724525964853e-05,
    'num_train_epochs': 2,
    'seed': 322,
    'gradient_accumulation_steps': 32
    }

### section 2

from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
dataset = load_dataset('glue', 'mrpc')
metric = load_metric('glue', 'mrpc')

def encode(examples):
    outputs = tokenizer(
        examples['sentence1'], examples['sentence2'], truncation=True)
    return outputs
encoded_dataset = dataset.map(encode, batched=True)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', return_dict=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = TrainingArguments(
    "test", evaluate_during_training=True, eval_steps=500, disable_tqdm=True)
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

# Defaut objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
trainer.hyperparameter_search(
    direction="maximize", 
    backend="ray", 
    n_samples=10, # number of trials
    # n_jobs=2  # number of parallel jobs, if multiple GPUs
)


# You can also easily swap different parameter tuning algorithms such as
# HyperBand, Bayesian Optimization, Population-Based Training:
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    # Choose among many libraries:
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    search_alg=HyperOptSearch(),
    # Choose among schedulers:
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    scheduler=AsyncHyperBand())


# Leveraging multiple GPUs for a parallel hyperparameter search
# is as easy as setting a setting a single argument too:
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

# You can also do distributed hyperparameter tuning by
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_jobs=4,  # number of parallel jobs, if multiple GPUs
    n_trials=4,  # number of hyperparameter samples
    # Aggressive termination of trials
    scheduler=AsyncHyperBand())