"""File for training XML-RoBERTa model."""

import os
from typing import Any, Literal
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.utils import resample
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
    dataset_path = f"{base_path}/data/processed"
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


def get_dataset(
    base_path: str,
    lang: Literal["en", "nl", "ar", "es", "all"],
    sample: bool = True,
    n_samples: int = 7000,
) -> tuple[Dataset, Dataset, Dataset]:

    def get_folder(
        lang: str, files: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        folder = f"processed_CT24_checkworthy_{files['folder']}"
        train = f"processed_{files['train']}"
        test = f"processed_{files['test']}"
        dev = f"processed_{files['dev']}"

        train_df = pd.read_csv(f"{base_path}/{folder}/{train}", sep="\t")
        test_df = pd.read_csv(f"{base_path}/{folder}/{test}", sep="\t")
        dev_df = pd.read_csv(f"{base_path}/{folder}/{dev}", sep="\t")

        return train_df, test_df, dev_df

    def resample_to_fixed_number(df, n_samples=5000):
        ones = df[df["class_label"] == "Yes"]
        zeros = df[df["class_label"] == "No"]
        print(len(ones), len(zeros))
        sets = []
        for dset in [ones, zeros]:
            if len(dset) < n_samples // 2:
                sets.append(
                    resample(
                        dset, replace=True, n_samples=n_samples // 2, random_state=567
                    )
                )
            else:
                sets.append(
                    resample(
                        dset, replace=False, n_samples=n_samples // 2, random_state=567
                    )
                )
        return pd.concat(sets)

    langs = {
        "en": {
            "train": "train.tsv",
            "test": "dev.tsv",
            "dev": "dev_test.tsv",
            "folder": "english",
        },
        "nl": {
            "train": "dutch_train.tsv",
            "test": "dutch_dev.tsv",
            "dev": "dutch_dev_test.tsv",
            "folder": "dutch",
        },
        "ar": {
            "train": "arabic_train.tsv",
            "test": "arabic_dev.tsv",
            "dev": "arabic_dev_test.tsv",
            "folder": "arabic",
        },
        "es": {
            "train": "spanish_train.tsv",
            "test": "spanish_dev.tsv",
            "dev": "spanish_dev_test.tsv",
            "folder": "spanish",
        },
    }

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_dev = pd.DataFrame()
    if lang == "all":
        for lang, files in langs.items():
            train, test, dev = get_folder(lang, files)

            if sample:
                df_train = resample_to_fixed_number(train, n_samples)
                df_test = resample_to_fixed_number(test, 500)
                df_dev = resample_to_fixed_number(dev, 500)

            df_train = pd.concat([df_train, train])
            df_test = pd.concat([df_test, test])
            df_dev = pd.concat([df_dev, dev])

    else:
        df_train, df_test, df_dev = get_folder(lang, langs[lang])
        if sample:
            df_train = resample_to_fixed_number(df_train, n_samples)
            df_test = resample_to_fixed_number(df_test, 500)
            df_dev = resample_to_fixed_number(df_dev, 500)

    df_train = df_train.rename(columns={"class_label": "labels", "tweet_text": "text"})
    df_test = df_test.rename(columns={"class_label": "labels", "tweet_text": "text"})
    df_dev = df_dev.rename(columns={"class_label": "labels", "tweet_text": "text"})

    return (
        Dataset.from_pandas(df_train),
        Dataset.from_pandas(df_test),
        Dataset.from_pandas(df_dev),
    )


def train(config=None):
    with wandb.init(config=config, project="dat550_project_lang") as run:
        save = True
        config = run.config
        # base_path = (
        #     "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text"
        # )
        base_path = "/home/emrds/repos/Multilingual-Check-worthiness-Estimation-in-Text"
        dataset_path, save_path = get_paths(base_path=base_path)

        train, test, dev_test = get_dataset(
            base_path=dataset_path, lang="en", sample=True, n_samples=7000
        )
        tokenized_train = train.map(tokenize_function, batched=True)
        tokenized_test = test.map(tokenize_function, batched=True)

        # optimizer = None
        # if config.optimizer and config.optimizer == "adam":
        #     optimizer = torch.optim.AdamW(
        #         model.parameters(),
        #         lr=config.learning_rate,
        #         weight_decay=config.weight_decay,
        #     )

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
            # optimizers=(optimizer if optimizer else None, None),
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
        "parameters": {
            "optimizer": {
                "values": ["adam", None],
            },
            "learning_rate": {
                "min": 1e-8,
                "max": 1e-4,
                "distribution": "log_uniform",
            },
            "num_train_epochs": {
                "values": [2, 3],
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

    config = {
        "learning_rate": 1e-6,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "weight_decay": 0.1,
        "warmup_steps": 500,
        "optimizer": "adam",
    }
    # sweep_id = wandb.sweep(sweep_config, project="factcheckworthiness")
    # wandb.agent(sweep_id, train, count=5)
    train(config=config)
