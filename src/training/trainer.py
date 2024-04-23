"""File for training XML-RoBERTa model."""

import os
from typing import Any
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.trainer_utils import EvalPrediction
import wandb


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-large", num_labels=2
)


def get_paths(base_path: str = "../../") -> Any:
    dataset_path = f"{base_path}/data/processed/"
    save_folder = f"{base_path}/results"

    folders = os.listdir(save_folder)
    run_numbers = [int(folder[3:]) for folder in folders if folder.startswith("run")]
    run_id = max(run_numbers, default=0) + 1
    save_path = f"{save_folder}/run{run_id}"
    os.makedirs(save_path)

    return dataset_path, save_path


def load_dataset(path: str) -> Dataset:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    return Dataset.from_pandas(df)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


def compute_metrics(p: EvalPrediction) -> dict:
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="binary"
    )
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train(config=None):
    with wandb.init():
        save = False
        config = wandb.config
        base_path = (
            "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/"
        )
        dataset_path, save_path = get_paths(base_path=base_path)

        train = f"{dataset_path}/merged_train.tsv"
        test = f"{dataset_path}/merged_test.tsv"
        tokenized_train = load_dataset(train).map(tokenize_function, batched=True)
        tokenized_test = load_dataset(test).map(tokenize_function, batched=True)

        optimizer = None

        if config.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        training_args = TrainingArguments(
            output_dir=save_path,
            logging_dir=f"{save_path}/logs",
            report_to="wandb",
            evaluation_strategy="epoch",
            eval_steps=0.1,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=compute_metrics,
            optimizers=(optimizer if optimizer else None, None),
        )
        trainer.train()
        if save:
            trainer.save_model()
            tokenizer.save_pretrained(save_path)


if __name__ == "__main__":

    sweep_config = {
        "method": "random",
        "metric": {
            "name": "f1",
            "goal": "maximize",
        },
        "optimizer": {
            "values": ["adam", None],
        },
        "parameters": {
            "learning_rate": {
                "min": 1e-8,
                "max": 1e-4,
                "distribution": "log_uniform",
            },
            "num_train_epochs": {
                "values": [2, 4],
            },
            "per_device_train_batch_size": {
                "values": [16, 32],
            },
            "per_device_eval_batch_size": {
                "values": [16, 32],
            },
            "weight_decay": {
                "min": 0.01,
                "max": 0.2,
                "distribution": "log_uniform",
            },
            "warmup_steps": {
                "values": [500, 1000],
            },
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="factcheckworthiness")
    wandb.agent(sweep_id, train, count=5)
