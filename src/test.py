#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
import json
import warnings
import argparse
import random
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import Dataset
from huggingface_hub import login

from helper_functions import (
    download_nltk_data,
    compute_metrics,
    preprocess_test_function,
    set_global_tokenizer_and_config
)

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test fine-tuned Seq2Seq models (BART or T5, with or without LoRA).")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["bart", "t5"],
        help="Type of base model architecture ('bart' for mBART/M2M100, 't5' for T5-like models)."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        required=True,
        help="Hugging Face Hub ID of the fine-tuned model or LoRA adapter."
    )
    parser.add_argument(
        "--is_lora_adapter",
        action="store_true",
        help="Flag if the hub_model_id points to LoRA adapters (requires PEFT)."
    )
    parser.add_argument(
        "--base_model_name_for_lora",
        type=str,
        default=None,
        help="Required if --is_lora_adapter is True and PeftConfig cannot infer base model. "
             "Hub ID of the base model the LoRA adapters were trained on."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/content/drive/MyDrive/Thesis_Data/test.tsv",
        help="Path to the test data TSV file."
    )
    parser.add_argument(
        "--results_dir_base",
        type=str,
        default="/content/drive/MyDrive/test_results_harmonized",
        help="Base directory to save test results and predictions."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for private models/repos."
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None, 
        help="Number of test samples to use for evaluation (None for all). This also limits the pool for random sample logging."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Max sequence length for tokenizer (should match training)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for generation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search generation."
    )
    parser.add_argument(
        "--num_samples_to_log",
        type=int,
        default=10,
        help="Number of random input/output examples to generate and save."
    )
    parser.add_argument(
        "--source_lang_code",
        type=str,
        default="ha",
        help="Source language code for mBART/M2M100 tokenizers (e.g., 'ha', 'en_XX')."
    )
    parser.add_argument(
        "--target_lang_code",
        type=str,
        default="ha",
        help="Target language code for mBART/M2M100 tokenizers (e.g., 'ha', 'en_XX')."
    )
    parser.add_argument(
        "--mount_drive",
        action="store_true",
        help="Mount Google Drive (for Colab)."
    )
    parser.add_argument(
        "--input_column_name", type=str, default="Noisy", help="Name of the input text column in the TSV file."
    )
    parser.add_argument(
        "--target_column_name", type=str, default="Clean", help="Name of the target/reference text column in the TSV file."
    )

    args = parser.parse_args()

    if args.is_lora_adapter and not PEFT_AVAILABLE:
        raise ImportError("PEFT library is not installed, but --is_lora_adapter was specified. Please install peft: pip install peft")
    if args.is_lora_adapter and not args.base_model_name_for_lora:
        print("Warning: --is_lora_adapter is True, but --base_model_name_for_lora is not set. "
              "Attempting to infer base model from PeftConfig. If this fails, please specify it.")
    return args

def main():
    args = parse_args()

    if args.mount_drive:
        try:
            from google.colab import drive
            if not os.path.exists('/content/drive/MyDrive'):
                drive.mount('/content/drive')
                print("Google Drive mounted successfully.")
            else:
                print("Google Drive already mounted.")
        except ImportError:
            print("Google Colab 'drive' module not found. Skipping drive mount.")

    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Using the `WANDB_DISABLED` environment variable is deprecated")
    warnings.filterwarnings("ignore", message=".*Neither `max_length`.*")

    if args.hf_token:
        try:
            login(token=args.hf_token)
            print("Logged in to Hugging Face Hub using provided token.")
        except Exception as e:
            print(f"Hugging Face login failed: {e}. Proceeding without explicit login.")

    # --- Directories and Paths ---
    model_name_for_path = args.hub_model_id.split('/')[-1]
    if args.is_lora_adapter:
        model_name_for_path += "-lora_eval"
    else:
        model_name_for_path += "-full_eval"

    current_test_results_dir = os.path.join(args.results_dir_base, model_name_for_path)
    os.makedirs(current_test_results_dir, exist_ok=True)
    print(f"Test results will be saved to: {current_test_results_dir}")

    download_nltk_data()

    print("--- Loading Test Data ---")
    if not os.path.exists(args.test_file):
        print(f"ERROR: Test file not found at {args.test_file}"); exit(1)
    try:
        test_df_full = pd.read_csv(args.test_file, sep='\t', on_bad_lines='warn')
        print(f"Loaded {len(test_df_full)} total samples from local file.")

        test_df_eval = test_df_full.copy()
        if args.test_size is not None and args.test_size > 0 and args.test_size < len(test_df_eval):
            print(f"Limiting evaluation data to {args.test_size} samples (randomly sampled).")
            test_df_eval = test_df_eval.sample(n=args.test_size, random_state=42).reset_index(drop=True)
        elif args.test_size is not None and args.test_size <= 0 :
             print(f"Warning: test_size ({args.test_size}) for evaluation is not positive. Using full dataset for evaluation.")

        required_columns = [args.input_column_name, args.target_column_name]
        if not all(col in test_df_eval.columns for col in required_columns):
            raise ValueError(f"Test data for evaluation missing required columns: {required_columns}. Found: {list(test_df_eval.columns)}")

        initial_eval_count = len(test_df_eval)
        test_df_eval = test_df_eval.dropna(subset=required_columns).reset_index(drop=True)
        cleaned_eval_count = len(test_df_eval)

        if cleaned_eval_count < initial_eval_count:
            print(f"Removed {initial_eval_count - cleaned_eval_count} rows with missing values from evaluation data.")
        if cleaned_eval_count == 0:
            raise ValueError("Evaluation dataset is empty after cleaning or due to test_size.")
        
        print(f"Using {len(test_df_eval)} samples for evaluation metrics.")
        # This dataset is used for trainer.evaluate()
        eval_dataset_hf = Dataset.from_pandas(test_df_eval)
        test_dataset_for_sampling_hf = Dataset.from_pandas(
            test_df_full.dropna(subset=required_columns).reset_index(drop=True)
        )


    except Exception as e:
        print(f"Error loading test data: {e}"); raise

    print("\n--- Loading Model and Tokenizer ---")
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Hub Model ID: {args.hub_model_id}")
    print(f"Is LoRA Adapter: {args.is_lora_adapter}")

    tokenizer_load_args = {"token": args.hf_token}
    model_load_args = {"token": args.hf_token}
    base_model_name_resolved = args.base_model_name_for_lora

    if args.model_type == "bart":
        tokenizer_load_args['src_lang'] = args.source_lang_code
        tokenizer_load_args['tgt_lang'] = args.target_lang_code
        print(f"BART model: src_lang='{args.source_lang_code}', tgt_lang='{args.target_lang_code}'")

    try:
        if args.is_lora_adapter:
            if not base_model_name_resolved:
                print(f"Attempting to infer base model name from PeftConfig at {args.hub_model_id}...")
                peft_config = PeftConfig.from_pretrained(args.hub_model_id, token=args.hf_token)
                base_model_name_resolved = peft_config.base_model_name_or_path
                print(f"Inferred base model: {base_model_name_resolved}")
            if not base_model_name_resolved:
                 raise ValueError("Could not determine base model for LoRA. Please specify --base_model_name_for_lora.")

            try:
                tokenizer = AutoTokenizer.from_pretrained(args.hub_model_id, **tokenizer_load_args)
                print(f"Loaded tokenizer from LoRA adapter repository: {args.hub_model_id}")
            except Exception:
                print(f"Could not load tokenizer from LoRA repo {args.hub_model_id}. Attempting to load from base model: {base_model_name_resolved}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name_resolved, **tokenizer_load_args)

            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_resolved, **model_load_args)
            model = PeftModel.from_pretrained(base_model, args.hub_model_id, token=args.hf_token)
            print("LoRA model loaded successfully.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.hub_model_id, **tokenizer_load_args)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.hub_model_id, **model_load_args)
            print("Full model loaded successfully.")

        set_global_tokenizer_and_config(tokenizer, args.max_seq_length)

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded onto device: {device}")

        if hasattr(model, 'generation_config'):
            generation_config = model.generation_config
        else:
            generation_config = GenerationConfig.from_model_config(model.config)

        generation_config.max_new_tokens = args.max_new_tokens
        generation_config.num_beams = args.num_beams

        if args.model_type == "bart":
            try:
                if hasattr(tokenizer, 'get_lang_id'):
                    forced_bos_id = tokenizer.get_lang_id(args.target_lang_code)
                elif hasattr(tokenizer, 'lang_code_to_id') and args.target_lang_code in tokenizer.lang_code_to_id:
                    forced_bos_id = tokenizer.lang_code_to_id[args.target_lang_code]
                else:
                    forced_bos_id = None
                    print(f"Warning: Could not determine forced_bos_token_id method for lang '{args.target_lang_code}'.")

                if forced_bos_id is not None:
                    generation_config.forced_bos_token_id = forced_bos_id
                    print(f"Set forced_bos_token_id: {forced_bos_id} for target '{args.target_lang_code}'")
            except Exception as e_gc:
                print(f"Warning: Could not set forced_bos_token_id: {e_gc}")
        model.generation_config = generation_config
        print(f"Generation config: max_new_tokens={model.generation_config.max_new_tokens}, num_beams={model.generation_config.num_beams}")

    except Exception as e:
        print(f"Error loading model/tokenizer: {e}"); raise

    print("\n--- Tokenizing Evaluation Dataset ---")
    try:
        tokenized_eval_dataset = eval_dataset_hf.map(
            lambda ex: preprocess_test_function(ex, model_type_for_preprocessing=args.model_type, input_column=args.input_column_name, target_column=args.target_column_name),
            batched=True,
            remove_columns=eval_dataset_hf.column_names
        )
        print("Evaluation data tokenized.")
    except Exception as e:
        print(f"Error tokenizing evaluation data: {e}"); raise

    print("\n--- Setting Up Evaluation using Seq2SeqTrainer ---")
    eval_training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(current_test_results_dir, "temp_trainer_output"),
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_config=model.generation_config,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(current_test_results_dir, 'temp_logs'),
        disable_tqdm=False,
        report_to=[],
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_training_args,
        eval_dataset=tokenized_eval_dataset, # Use the tokenized evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Trainer initialized for evaluation.")

    print("\n--- Running Evaluation ---")
    evaluation_successful = False
    try:
        eval_results = trainer.evaluate() # Evaluates on tokenized_eval_dataset
        print("\n--- Evaluation Metrics ---")
        metrics_to_save = {k: v for k, v in eval_results.items() if k.startswith('eval_')}
        print(json.dumps(metrics_to_save, indent=2))
        metrics_file_path = os.path.join(current_test_results_dir, "evaluation_metrics.json")
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"Evaluation metrics saved to: {metrics_file_path}")
        evaluation_successful = True
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

    # --- Generating and Logging Sample Predictions ---
    if evaluation_successful: # Proceed only if evaluation itself didn't crash
        print("\n--- Generating and Logging Sample Predictions ---")
        num_samples_to_actually_log = min(args.num_samples_to_log, len(test_dataset_for_sampling_hf))

        if num_samples_to_actually_log == 0:
            print("No samples available in the (cleaned) test dataset to log.")
        else:
            sample_output_file = os.path.join(current_test_results_dir, "sample_predictions_log.txt")
            print(f"Generating and logging {num_samples_to_actually_log} random samples to: {sample_output_file}")
            
            # Ensure we don't try to sample more than available
            random_indices = random.sample(range(len(test_dataset_for_sampling_hf)), num_samples_to_actually_log)
            
            try:
                with open(sample_output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model: {args.hub_model_id} ({args.model_type}, LoRA: {args.is_lora_adapter})\n")
                    f.write(f"Test File Used for Sampling Pool: {args.test_file}\n")
                    f.write(f"Total Samples in Sampling Pool (after cleaning): {len(test_dataset_for_sampling_hf)}\n")
                    f.write(f"Number of Samples Logged: {num_samples_to_actually_log}\n")
                    f.write("="*50 + "\n\n")

                    for i, idx in enumerate(tqdm(random_indices, desc="Generating Samples for Log")):
                        sample_data = test_dataset_for_sampling_hf[idx]
                        input_text = sample_data[args.input_column_name]
                        reference_text = sample_data[args.target_column_name]
                        
                        # Tokenize and predict this single sample
                        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(device)
                        try:
                            with torch.no_grad():
                                outputs = model.generate(inputs.input_ids, generation_config=model.generation_config)
                            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        except Exception as gen_e:
                            prediction = f"Error during generation for sample: {gen_e}"
                        
                        f.write(f"--- Sample {i+1} (Original Index in Sampling Pool: {idx}) ---\n")
                        f.write(f"Input ({args.input_column_name}):     {input_text}\n")
                        f.write(f"Prediction:        {prediction.strip()}\n")
                        f.write(f"Reference ({args.target_column_name}): {reference_text}\n\n")
                print(f"Sample predictions logged to {sample_output_file}")
            except Exception as sample_e:
                print(f"Error during sample logging: {sample_e}")
                import traceback
                traceback.print_exc()
    else:
        print("\nSkipping sample prediction logging due to evaluation error or no samples to log.")

    print(f"\n--- Evaluation Script Finished for {args.hub_model_id} ---")

if __name__ == "__main__":
    main()
