from datasets import ClassLabel, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import wandb
import random
import torch

dataset_en = load_dataset("iai-group/clef2024_checkthat_task1_en")
dataset_en = dataset_en.rename_column("Text", "text")
dataset_en = dataset_en.rename_column("class_label", "label")
dataset_en = dataset_en.rename_column("Sentence_id", "id")
full_dataset_train = dataset_en["train"]
full_dataset_test = dataset_en["test"]
# Load the IMDb dataset
for lang in ["es", "ar", "nl"]:
    dataset = load_dataset(f"iai-group/clef2024_checkthat_task1_{lang}")
    dataset = dataset.rename_column("tweet_text", "text")
    dataset = dataset.rename_column("class_label", "label")
    dataset = dataset.rename_column("tweet_id", "id")
    dataset = dataset.remove_columns("tweet_url")
    full_dataset_train = concatenate_datasets([full_dataset_train, dataset["train"]])
    full_dataset_test = concatenate_datasets([full_dataset_test, dataset["test"]])
# Define features with ClassLabel for label encoding
label_feature = ClassLabel(names=["Yes", "No"])
features = full_dataset_train.features.copy()

features["label"] = label_feature

# Map string labels to integers
full_dataset_train = full_dataset_train.cast(features)
full_dataset_test = full_dataset_test.cast(features)
print("Train Dataset:")
print(full_dataset_train)
print("\nTest Dataset:")
print(full_dataset_test)
wandb.init(project="clef24_task1", entity="vinays")
# print(dataset)
# total_samples = len(full_dataset_train)
# random_indices = random.sample(range(total_samples), 1000)
# train_dataset = full_dataset_train.select(random_indices)
train_dataset = full_dataset_train
print(train_dataset.shape)
test_dataset = full_dataset_test
print(test_dataset.shape)
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-large", num_labels=2
)  # 2 labels for binary classification
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")


# Preprocess the data
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=256
    )


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

print(tokenized_train_dataset[0])


# Compute metrics function for evaluation
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="wandb",  # Enable logging to wandb
    evaluation_strategy="epoch",
)


# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,  # Pass the compute_metrics function
)

# Train the model
trainer.train()

# Save the model
model_path = "./bert-imdb"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
# Optionally, you can finish the wandb run when training is done
wandb.finish()
