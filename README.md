# Multilingual-Check-worthiness-Estimation-in-Text  📚🔍🔎

# Fact-Checking Task README

## Overview
This repository is dedicated to a task aimed at evaluating whether a statement, derived from either a tweet or a political debate, warrants fact-checking. The decision-making process involves analyzing whether the statement contains a verifiable factual claim and assessing its potential harm before assigning a final label for its check-worthiness.

## Task Description
The primary goal of this task is to assess the veracity of statements sourced from tweets or political debates. Key considerations include:
- Determining whether the statement includes a verifiable factual claim.
- Evaluating the potential harm associated with the statement.
- Assigning a label indicating the statement's check-worthiness.

## Dataset
The dataset for this task is available at the following location:
[Task 1 Dataset](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/blob/main/task1/data/CT24_checkworthy_english.zip)

## Model Training
Transformer models, particularly XLM-RoBERTa-Large, are trained using the HuggingFace library. The training process involves utilizing the provided train split of the dataset.

## Model Evaluation
Once trained, the models are evaluated using the dev-test split of the dataset.

## Baselines
As baselines, we employ GPT-4 or open-source models such as Llama2 or Mistral for predicting the check-worthiness of statements.

## Repository Structure
- **data/**: Contains the dataset split into training, validation, and testing sets.
- **models/**: Stores pre-trained and fine-tuned transformer models.
- **notebooks/**: Includes Jupyter notebooks for data preprocessing, model training, and evaluation.
- **src/**: Houses source code for data preprocessing, model implementation, and evaluation.
- **requirements.txt**: Lists the required Python packages and their versions.

## Instructions
1. Clone this repository.
2. Install the necessary dependencies listed in `requirements.txt`.
3. Preprocess the data if required.
4. Train the transformer models using the provided training split.
5. Evaluate the trained models on the dev-test split.
6. Experiment with different baselines and compare their performance.


## Acknowledgments
We acknowledge the organizers of the task and the contributors to the dataset.

## References
- HuggingFace: [https://huggingface.co/](https://huggingface.co/)
- Dataset: [Link to the dataset](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1)
- GPT-4: [Provide relevant link if available]
- Llama2: [Provide relevant link if available]
- Mistral: [Provide relevant link if available]


