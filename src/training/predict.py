import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

en_dataset_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/data/processed/processed_CT24_checkworthy_arabic/processed_arabic_dev_test.tsv"
du_dataset_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/data/processed/processed_CT24_checkworthy_dutch/processed_dutch_dev_test.tsv"
es_dataset_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/data/processed/processed_CT24_checkworthy_spanish/processed_spanish_dev_test.tsv"
ar_dataset_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/data/processed/processed_CT24_checkworthy_arabic/processed_arabic_dev_test.tsv"

en_model_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/results/run1/"
es_model_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/results/run2/"
du_model_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/results/run3/"
ar_model_path = "/home/stud/emartin/bhome/Multilingual-Check-worthiness-Estimation-in-Text/results/run4/"


def load_dataset(path: str) -> Dataset:
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.rename(columns={"class_label": "label", "tweet_text": "text"})
        df["label"] = df["label"].apply(lambda x: 1 if x == "Yes" else 0)
        df = df.drop("tweet_id", axis=1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    return Dataset.from_pandas(df)


def compute_metrics(p: EvalPrediction) -> dict:
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="binary"
    )
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def predict(model_path: str, dataset_path: str, lang: str):

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    dataset = load_dataset(dataset_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    dataset = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model, compute_metrics=compute_metrics, eval_dataset=dataset
    )
    trainer.evaluate()
    predictions = trainer.predict(dataset)
    print(f"Results for {lang}:\n{predictions.metrics}")


if __name__ == "__main__":
    datasets = [
        ("en", en_dataset_path, en_model_path),
        ("es", es_dataset_path, es_model_path),
        ("nl", du_dataset_path, du_model_path),
        ("ar", ar_dataset_path, ar_model_path),
    ]
    for lang, dataset, model in datasets:
        predict(model, dataset, lang)
