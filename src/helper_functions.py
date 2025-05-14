import os
import nltk
import numpy as np
import editdistance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from jiwer import wer

tokenizer_instance = None
g_max_seq_length = None 

def set_global_tokenizer_and_config(tokenizer, max_seq_length_val):
    """Sets the global tokenizer instance and max_seq_length."""
    global tokenizer_instance, g_max_seq_length
    tokenizer_instance = tokenizer
    g_max_seq_length = max_seq_length_val

def set_global_tokenizer(tokenizer):
    """Sets the global tokenizer instance."""
    global tokenizer_instance
    tokenizer_instance = tokenizer
    
def download_nltk_data():
    """Downloads necessary NLTK data if not found."""
    nltk_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    for resource_path, resource_name in nltk_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK '{resource_name}' data...")
            nltk.download(resource_name, quiet=True)

def compute_bleu(pred, label):
    smoothie = SmoothingFunction().method4
    try:
        if not label: return 0.0
        ref_tokens = [label.split()]
        pred_tokens = pred.split()
        if not pred_tokens: return 0.0
        max_n = min(len(ref_tokens[0]), 4)
        if max_n == 0: return 0.0
        weights = [1.0/max_n] * max_n
        return sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothie)
    except Exception:
        return 0.0

def compute_token_f1(pred, label):
    pred_tokens = set(pred.split())
    label_tokens = set(label.split())
    common = pred_tokens & label_tokens
    if not common or len(pred_tokens) == 0 or len(label_tokens) == 0: return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(label_tokens)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def compute_wer(pred, label):
    try:
        return wer(label, pred)
    except Exception:
        return 1.0

def compute_cer(pred, label):
    try:
        if not label: return 1.0 if pred else 0.0
        return editdistance.eval(pred, label) / len(label)
    except ZeroDivisionError:
        return 1.0 if pred else 0.0
    except Exception:
        return 1.0

def compute_meteor(pred, label):
    try:
        if not label: return 0.0
        return meteor_score.meteor_score([label.split()], pred.split())
    except Exception:
        return 0.0

def compute_metrics(eval_preds):
    global tokenizer_instance
    if tokenizer_instance is None:
        raise ValueError("Global tokenizer must be set before calling compute_metrics.")

    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer_instance.pad_token_id)

    decoded_preds = tokenizer_instance.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer_instance.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    metrics_scores = {"bleu": [], "f1": [], "wer": [], "cer": [], "meteor": []}
    for pred, label in zip(decoded_preds, decoded_labels):
        if not label: continue
        metrics_scores["bleu"].append(compute_bleu(pred, label))
        metrics_scores["f1"].append(compute_token_f1(pred, label))
        metrics_scores["wer"].append(compute_wer(pred, label))
        metrics_scores["cer"].append(compute_cer(pred, label))
        metrics_scores["meteor"].append(compute_meteor(pred, label))

    result = {
        "eval_bleu": np.mean(metrics_scores["bleu"]) if metrics_scores["bleu"] else 0.0,
        "eval_f1": np.mean(metrics_scores["f1"]) if metrics_scores["f1"] else 0.0,
        "eval_wer": np.mean(metrics_scores["wer"]) if metrics_scores["wer"] else 1.0,
        "eval_cer": np.mean(metrics_scores["cer"]) if metrics_scores["cer"] else 1.0,
        "eval_meteor": np.mean(metrics_scores["meteor"]) if metrics_scores["meteor"] else 0.0
    }
    return result

def preprocess_test_function(examples, model_type_for_preprocessing, input_column='Noisy', target_column='Clean'):
    """Tokenizes the input and target texts for testing, using global tokenizer and max_length."""
    global tokenizer_instance, g_max_seq_length
    if tokenizer_instance is None or g_max_seq_length is None:
        raise ValueError("Global tokenizer and max_seq_length must be set before calling preprocess_test_function.")

    if input_column not in examples or target_column not in examples:
        available_columns = list(examples.keys()) if isinstance(examples, dict) else []
        if not available_columns and isinstance(examples, list) and examples and isinstance(examples[0], dict):
            available_columns = list(examples[0].keys())
        raise ValueError(f"Dataset must contain '{input_column}' and '{target_column}'. Found: {available_columns}")

    noisy_texts = [str(text) if text is not None else "" for text in examples[input_column]]
    clean_texts = [str(text) if text is not None else "" for text in examples[target_column]]

    model_inputs = tokenizer_instance(noisy_texts, padding="max_length", truncation=True, max_length=g_max_seq_length)

    if model_type_for_preprocessing == "bart":
        try:
            with tokenizer_instance.as_target_tokenizer():
                labels = tokenizer_instance(clean_texts, padding="max_length", truncation=True, max_length=g_max_seq_length)
        except Exception:
            print("Note: tokenizer.as_target_tokenizer() context manager might not be fully effective or needed. Proceeding with standard tokenization for labels.")
            labels = tokenizer_instance(clean_texts, padding="max_length", truncation=True, max_length=g_max_seq_length)
    else: # T5 and other models
        labels = tokenizer_instance(clean_texts, padding="max_length", truncation=True, max_length=g_max_seq_length)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function(examples, max_seq_length, model_type):
    """Tokenizes the input and target texts."""
    global tokenizer_instance
    if tokenizer_instance is None:
        raise ValueError("Global tokenizer must be set via set_global_tokenizer() before calling preprocess_function.")

    noisy_texts = [str(text) if text is not None else "" for text in examples['Noisy']]
    clean_texts = [str(text) if text is not None else "" for text in examples['Clean']]

    # For mBART/M2M100, tokenizer uses src_lang for inputs
    inputs = tokenizer_instance(noisy_texts, padding="max_length", truncation=True, max_length=max_seq_length)

    if model_type == "bart":
        with tokenizer_instance.as_target_tokenizer():
            targets = tokenizer_instance(clean_texts, padding="max_length", truncation=True, max_length=max_seq_length)
    else: # T5 and other models
        targets = tokenizer_instance(clean_texts, padding="max_length", truncation=True, max_length=max_seq_length)


    inputs['labels'] = targets['input_ids']
    return inputs