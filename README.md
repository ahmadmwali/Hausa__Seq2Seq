This project contains a suite of Python scripts for data processing, model training, evaluation, and prediction, primarily focused on sequence-to-sequence tasks (like text harmonization/correction) and text classification (sentiment analysis).

## Table of Contents

- [Project Overview](#project-overview)
- [File Descriptions](#file-descriptions)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Data Preparation (`get_data.py`)](#1-data-preparation-get_datapy)
  - [2. Training a Seq2Seq Model (`train.py`)](#2-training-a-seq2seq-model-trainpy)
  - [3. Evaluating a Seq2Seq Model (`test.py`)](#3-evaluating-a-seq2seq-model-testpy)
  - [4. Predicting with a Seq2Seq Model (`predict_on_dataframe.py`)](#4-predicting-with-a-seq2seq-model-predict_on_dataframepy)
  - [5. Evaluating a Downstream Classification Task (`evaluate_downstream.py`)](#5-evaluating-a-downstream-classification-task-evaluate_downstreampy)
- [Helper Functions](#helper-functions)
- [Requirements](#requirements)

## Project Overview

This project provides tools to:
1.  Prepare text data by merging, cleaning, segmenting, and applying noise to create datasets for tasks like text harmonization.
2.  Fine-tune sequence-to-sequence (Seq2Seq) models (e.g., mBART, T5) for tasks such as correcting noisy text to clean text, with support for LoRA.
3.  Evaluate the performance of these fine-tuned Seq2Seq models using various metrics (BLEU, WER, CER, METEOR, F1).
4.  Use the trained Seq2Seq models to generate predictions on new text data.
5.  Train and evaluate text classification models (e.g., for sentiment analysis) on datasets, potentially using the outputs of the harmonization models.

## File Descriptions

-   **`get_data.py`**: Merges, cleans, segments text from input files, applies various types of noise, and splits the data into training, validation, and test sets in TSV format.
-   **`train.py`**: Fine-tunes sequence-to-sequence models (like mBART or T5) using Hugging Face Transformers. It supports training with or without LoRA (Low-Rank Adaptation) for efficient fine-tuning. It loads data prepared by `get_data.py`, configures training arguments, handles tokenization, and saves the trained model.
-   **`test.py`**: Evaluates a fine-tuned sequence-to-sequence model (either a full model or LoRA adapters) from the Hugging Face Hub or a local path. It loads a test dataset, generates predictions, computes various metrics (BLEU, WER, CER, METEOR, F1), and logs sample predictions.
-   **`predict_on_dataframe.py`**: Takes a trained sequence-to-sequence model (full or LoRA) and a DataFrame (from a TSV/CSV file) as input. It generates predictions for a specified text column in the DataFrame and saves the DataFrame with an added column containing these predictions.
-   **`evaluate_downstream.py`**: Trains and evaluates a text classification model (e.g., for sentiment analysis) using Hugging Face Transformers. It can take a TSV file with text and labels, preprocess the data, train a classifier (like BERT-based models), and report classification metrics (accuracy, precision, recall, F1). It can be used to evaluate the impact of text harmonization on downstream tasks.
-   **`helper_functions.py`**: Contains utility functions used by other scripts, such as NLTK data download, metrics computation (BLEU, WER, CER, F1, METEOR), and data preprocessing/tokenization functions for Seq2Seq models.
-   **`requirements.txt`**: Lists the Python package dependencies for this project.

## Setup and Installation

1.  **Clone the repository (if applicable) or download the files.**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data (first time setup for `helper_functions.py`):**
    The `helper_functions.py` script will attempt to download necessary NLTK resources (`punkt`, `wordnet`, `omw-1.4`) if they are not found. This might require an internet connection.

## Usage

The scripts are designed to be run from the command line with various arguments. Below are typical workflows and example commands.

### 1. Data Preparation (`get_data.py`)

This script processes raw text files to create structured datasets for training Seq2Seq models.

**Functionality:**
-   Merges multiple input text files.
-   Cleans text (removes citations, extra spaces, etc.).
-   Segments text into sentences.
-   Applies configurable noise (random spacing, remove spaces, incorrect characters, delete characters, duplicate characters) to create 'Noisy' versions of 'Clean' sentences.
-   Splits the data into train, validation, and test sets.
-   Saves datasets in TSV format with 'Noisy' and 'Clean' columns.
-   Optionally excludes religious texts.

**Example:**
```bash
python get_data.py \
    --input_files path/to/your/file1.txt path/to/your/file2.txt \
    --output_dir_base ./Files_Processed \
    --train_size 80000 \
    --val_size 10000 \
    --test_size 10000 \
    --noise_random_spacing 0.02 \
    --seed 42
