import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.trainer_utils import EvalPrediction

from training.trainer import load_dataset


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

en_dataset_path = (
    "../data/processed/processed_CT24_checkworthy_arabic/processed_arabic_dev_test.tsv"
)
du_dataset_path = (
    "../data/processed/processed_CT24_checkworthy_dutch/processed_dutch_dev_test.tsv"
)
es_dataset_path = "../data/processed/processed_CT24_checkworthy_spanish/processed_spanish_dev_test.tsv"
ar_dataset_path = (
    "../data/processed/processed_CT24_checkworthy_arabic/processed_arabic_dev_test.tsv"
)

en_model_path = "../results/run1"
es_model_path = "../results/run2"
du_model_path = "../results/run3"
ar_model_path = "../results/run4"


datasets = [
    ("en", en_dataset_path, en_model_path),
    ("es", es_dataset_path, es_model_path),
    ("nl", du_dataset_path, du_model_path),
    ("ar", ar_dataset_path, ar_model_path),
]


def compute_metrics(p: EvalPrediction) -> dict:
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="binary"
    )
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


for lang, dataset_path, model_path in datasets.items():

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    dataset = load_dataset(dataset_path)

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    trainer.evaluate()
    predictions = trainer.predict(dataset)
