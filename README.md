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
[Task 1 Dataset](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1/data)

### Input Data Format (Unimodal - Text -- Tweets)
For Arabic, and Spanish we use the same data format in the train, dev and dev_test files. Each file is TAB seperated (TSV file) containing the tweets and their labels. The text encoding is UTF-8. Each row in the file has the following format:


**Definitions:**

- **tweet_id**: Unique ID assigned to the tweet by Twitter.
- **tweet_url**: URL link to the specific tweet.
- **tweet_text**: Actual content of the tweet.
- **class_label**: Categorical label indicating 'Yes' or 'No'.

### Examples:

- 1235648554338791427	https://twitter.com/A6Asap/status/1235648554338791427	COVID-19 health advice⚠️ https://t.co/XsSAo52Smu	No
- 1235287380292235264	https://twitter.com/ItsCeliaAu/status/1235287380292235264	There's not a single confirmed case of an Asian infected in NYC. Stop discriminating cause the virus definitely doesn't. #racist #coronavirus https://t.co/Wt1NPOuQdy	Yes
- 1236020820947931136	https://twitter.com/ddale8/status/1236020820947931136	Epidemiologist Marc Lipsitch, director of Harvard's Center for Communicable Disease Dynamics: “In the US it is the opposite of contained.' https://t.co/IPAPagz4Vs	Yes
- ... 

**Note** that the gold labels for the task are the ones in the class_label column.

### Input Data Format (Unimodal - Text -- Political debates)
For English we use the same data format in the train, dev and dev_test files. Each file is TAB seperated (TSV file) containing the tweets and their labels. The text encoding is UTF-8. Each row in the file has the following format:


**Definitions:**

- **sentence_id**: Unique ID assigned to the sentence from a political debate.
- **text**: The text of the sentence.
- **class_label**: Categorical label indicating 'Yes' or 'No'.

### Examples:

- 30313	And so I know that this campaign has caused some questioning and worries on the part of many leaders across the globe.	No
- 19099	"Now, let's balance the budget and protect Medicare, Medicaid, education and the environment."	No
- 33964	I'd like to mention one thing.	No
- ... 


**Note:** The gold labels for the task are found in the `class_label` column.


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
4. Train the transformer models using `src/training/train.py`.
5. Evaluate the trained models on the dev-test split using `src/training/predict.py`.
6. Experiment with different baselines and compare their performance.


## Acknowledgments
We acknowledge the organizers of the task and the contributors to the dataset.

## References
- HuggingFace: [https://huggingface.co/](https://huggingface.co/)
- Dataset: [Link to the dataset](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1)
- GPT-4: [Provide relevant link if available]
- Llama2: [Provide relevant link if available]
- Mistral: [Provide relevant link if available]


