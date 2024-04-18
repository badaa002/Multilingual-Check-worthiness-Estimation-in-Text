"""File for training XML-RoBERTa model."""

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import wandb
import os
from sklearn.utils import shuffle
from typing import Any


def get_paths(is_gdrive: bool = False) -> Any:
    """Get paths for training data and storing results.

    Args:
        is_gdrive: Whether to use Google Drive path for storing results and
            retrieving data.

    Returns:
        dataset_path: Path to training data.
        save_path: Path to store results.
    """
    dataset_path = (
        "./drive/MyDrive/data"
        if is_gdrive
        else "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/data/processed/processed_CT24_checkworthy_english"
    )
    save_folder = (
        "./drive/MyDrive/results"
        if is_gdrive
        else "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/results"
    )

    folders = os.listdir(save_folder)
    run_numbers = [int(folder[3:]) for folder in folders if folder.startswith("run")]
    run_id = max(run_numbers, default=0) + 1

    save_path = f"{save_folder}/run{run_id}"

    os.makedirs(save_path)
    return dataset_path, save_path


def load_dataset(path: str) -> Dataset:
    df = pd.read_csv(path, sep="\t")
    df = df.copy()
    df["label"] = df["class_label"].apply(lambda x: 1 if x == "Yes" else 0)
    df = df.drop(columns=["class_label", "sentence_id"])

    dataset = Dataset.from_pandas(df)
    return dataset


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


def compute_metrics(p: EvalPrediction) -> dict[str, Any]:
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    print(f"Preds: {preds}")
    print(f"Num ones: ", len([p for p in preds if p == 1]))
    print(f"Num zeroes: ", len([p for p in preds if p == 0]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train(
    save_path: str, tokenized_train_dataset: Dataset, tokenized_test_dataset: Dataset
):
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    training_args = TrainingArguments(
        output_dir=save_path,  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.05,  # strength of weight decay
        logging_steps=500,  # how many batches to run before saving a backup of the run
        evaluation_strategy="epoch",  # when to run the model evaluation (check what the model has learned agains the data it has trained on)
        report_to="wandb",  # where to upload the data
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    trainer.train()

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    wandb.finish()


if __name__ == "__main__":
    dataset_path, save_path = get_paths(is_gdrive=False)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    train_dataset = load_dataset(f"{dataset_path}/processed_train.tsv")
    test_dataset = load_dataset(f"{dataset_path}/processed_dev.tsv")
    dev_test_dataset = load_dataset(f"{dataset_path}/processed_dev_test.tsv")

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    tokenized_dev_test_dataset = dev_test_dataset.map(tokenize_function, batched=True)

    wandb.init(project="dat550_project_base_full")
    train(save_path, tokenized_train_dataset, tokenized_test_dataset)
