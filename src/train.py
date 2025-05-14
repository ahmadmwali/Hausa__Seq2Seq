#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
import json
import warnings
import argparse

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from datasets import Dataset
from huggingface_hub import login

# Import helper functions
from helper_functions import (
    download_nltk_data,
    compute_metrics,
    preprocess_function,
    set_global_tokenizer_and_config
)

# Conditional import for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Seq2Seq models with optional LoRA.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["bart", "t5"],
        help="Type of model to train ('bart' for mBART/M2M100, 't5' for T5-like models)."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="Name of the pretrained model from Hugging Face Model Hub (e.g., 'facebook/m2m100_418M' or 'castorini/afriteva_small')."
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Flag to enable LoRA training."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/content/drive/MyDrive/Thesis_Data/train.tsv",
        help="Path to the training data TSV file."
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="/content/drive/MyDrive/Thesis_Data/validation.tsv",
        help="Path to the validation data TSV file."
    )
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="/content/drive/MyDrive/results_harmonized",
        help="Base directory to save results and checkpoints."
    )
    parser.add_argument(
        "--hub_model_id_prefix",
        type=str,
        default="ahmadmwali/harmonized",
        help="Prefix for Hugging Face Hub model ID (e.g., 'your-username/your-model'). Will be appended with model details."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None, # Replace with your token or set as environment variable
        help="Hugging Face API token. If not provided, attempts login or uses cached token."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Flag to push the model to Hugging Face Hub."
    )

    # Data parameters
    parser.add_argument("--train_size", type=int, default=None, help="Number of training samples to use (None for all).")
    parser.add_argument("--val_size", type=int, default=None, help="Number of validation samples to use (None for all).")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length for tokenizer.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for generation during evaluation.")

    # mBART/M2M100 specific tokenizer params (only used if model_type is 'bart')
    parser.add_argument("--source_lang_code", type=str, default="ha", help="Source language code for mBART/M2M100 (e.g., 'ha', 'en_XX').")
    parser.add_argument("--target_lang_code", type=str, default="ha", help="Target language code for mBART/M2M100 (e.g., 'ha', 'en_XX').")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--lora_learning_rate", type=float, default=2e-4, help="Learning rate for LoRA training.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size per device.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay.")

    # LoRA specific parameters (only used if use_lora is True)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    # Note: LORA_TARGET_MODULES will be set based on model_type

    # Drive mount (Colab specific)
    parser.add_argument("--mount_drive", action="store_true", help="Mount Google Drive (for Colab).")


    args = parser.parse_args()

    if args.use_lora and not PEFT_AVAILABLE:
        raise ImportError("PEFT library is not installed, but --use_lora was specified. Please install peft: pip install peft")

    return args

def main():
    args = parse_args()

    # Mount Google Drive if in Colab and specified
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
            pass

    # Ignore specific warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Using the `WANDB_DISABLED` environment variable is deprecated")

    # --- Hugging Face Login ---
    if args.push_to_hub:
        if args.hf_token:
            login(token=args.hf_token)
            print("Logged in to Hugging Face Hub using provided token.")
        else:
            try:
                login() # Attempts interactive login or uses cached token
                print("Logged in to Hugging Face Hub.")
            except Exception as e:
                print(f"Hugging Face login failed: {e}. Push to Hub might fail or be disabled.")
                if args.push_to_hub:
                    print("Disabling push_to_hub due to login failure.")
                    args.push_to_hub = False


    # --- Directories and Paths ---
    os.makedirs(args.output_dir_base, exist_ok=True)
    model_suffix = f"{args.base_model_name.split('/')[-1]}"
    model_suffix += "-lora" if args.use_lora else "-full"
    model_output_path = os.path.join(args.output_dir_base, model_suffix)
    hub_model_id = f"{args.hub_model_id_prefix}-{args.model_type}-{model_suffix}" if args.push_to_hub else None

    print(f"Model Type: {args.model_type.upper()}")
    print(f"Base Model: {args.base_model_name}")
    print(f"Using LoRA: {args.use_lora}")
    print(f"Local results will be saved to: {model_output_path}")
    if args.push_to_hub and hub_model_id:
        print(f"Model will be pushed to Hub ID: {hub_model_id}")
    else:
        print("Pushing to Hugging Face Hub is disabled or Hub ID not generated.")

    # --- NLTK Data ---
    download_nltk_data()

    # --- Load Data ---
    print("--- Loading and Preparing Data ---")
    if not os.path.exists(args.train_file):
        print(f"ERROR: Training file not found at {args.train_file}"); exit(1)
    if not os.path.exists(args.val_file):
        print(f"ERROR: Validation file not found at {args.val_file}"); exit(1)

    try:
        train_df = pd.read_csv(args.train_file, sep='\t', on_bad_lines='warn')
        val_df = pd.read_csv(args.val_file, sep='\t', on_bad_lines='warn')
        print(f"Initial raw data: {len(train_df)} train, {len(val_df)} validation.")

        if args.train_size is not None: train_df = train_df.head(args.train_size)
        if args.val_size is not None: val_df = val_df.head(args.val_size)
        print(f"Using data: {len(train_df)} train, {len(val_df)} validation.")
    except Exception as e:
        print(f"Error during data loading: {e}"); exit(1)

    required_columns = ['Noisy', 'Clean']
    if not all(col in train_df.columns for col in required_columns) or \
       not all(col in val_df.columns for col in required_columns):
        print(f"Data missing required columns: {required_columns}. Found: Train {train_df.columns}, Val {val_df.columns}"); exit(1)

    train_df = train_df.dropna(subset=required_columns).reset_index(drop=True)
    val_df = val_df.dropna(subset=required_columns).reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        print("Error: No data after cleaning. Check files/sizes."); exit(1)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    print("\nTrain Dataset sample:", train_dataset[0])
    print("Validation Dataset sample:", val_dataset[0])

    # --- Initialize Tokenizer and Model ---
    print("\n--- Initializing Tokenizer and Model ---")
    tokenizer_args = {}
    if args.model_type == "bart":
        tokenizer_args['src_lang'] = args.source_lang_code
        tokenizer_args['tgt_lang'] = args.target_lang_code
        print(f"mBART/M2M100 Tokenizer: src_lang='{args.source_lang_code}', tgt_lang='{args.target_lang_code}'")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, **tokenizer_args)
    set_global_tokenizer_and_config(tokenizer, args.max_seq_length) # Set global tokenizer for helper functions

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name)

    # --- LoRA Configuration (if applicable) ---
    if args.use_lora:
        if not PEFT_AVAILABLE: # Should have been caught by argparser, but double check
             print("Error: PEFT library not available for LoRA. Exiting.")
             exit(1)
        print("\n--- Configuring LoRA ---")
        # Determine LORA_TARGET_MODULES based on model_type
        if args.model_type == "bart": # mBART, M2M100 are BART-like
            lora_target_modules = ["q_proj", "v_proj"] # Common for BART architectures
        elif args.model_type == "t5":
            lora_target_modules = ["q", "v"] # Common for T5 architectures
        else:
            print(f"Warning: LoRA target modules not pre-defined for model_type '{args.model_type}'. Using default T5 targets. This might need adjustment.")
            lora_target_modules = ["q", "v"]
        print(f"LoRA Target Modules: {lora_target_modules}")

        # Optional: Prepare model for k-bit training (e.g., for QLoRA or gradient checkpointing)
        # model = prepare_model_for_kbit_training(model) # Uncomment if using gradient_checkpointing=True or QLoRA

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, lora_config)
        print("LoRA model created successfully:")
        model.print_trainable_parameters()
        effective_learning_rate = args.lora_learning_rate
    else:
        effective_learning_rate = args.learning_rate

    # --- Generation Configuration ---
    try:
        generation_config = model.generation_config
    except AttributeError: # PeftModel might wrap it differently
        generation_config = model.base_model.model.generation_config

    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.early_stopping = True
    generation_config.num_beams = 4 # Default beam search

    if args.model_type == "bart": # mBART/M2M100 forced BOS token
        try:
            # M2M100 uses get_lang_id, mBART might use lang_code_to_id
            if hasattr(tokenizer, 'get_lang_id'):
                forced_bos_id = tokenizer.get_lang_id(args.target_lang_code)
            elif hasattr(tokenizer, 'lang_code_to_id') and args.target_lang_code in tokenizer.lang_code_to_id:
                 forced_bos_id = tokenizer.lang_code_to_id[args.target_lang_code]
            else:
                forced_bos_id = None
                print(f"Warning: Could not determine forced_bos_token_id mechanism for {args.base_model_name} and lang '{args.target_lang_code}'.")

            if forced_bos_id is not None:
                generation_config.forced_bos_token_id = forced_bos_id
                print(f"Set forced_bos_token_id to: {forced_bos_id} for language '{args.target_lang_code}'")
        except Exception as e:
            print(f"Warning: Could not set forced_bos_token_id for '{args.target_lang_code}'. Error: {e}")
            print(f"Ensure '{args.target_lang_code}' is valid for {args.base_model_name}.")

    model.generation_config = generation_config # Attach updated config
    print(f"Generation: Max New Tokens: {args.max_new_tokens}, Beams: {generation_config.num_beams}")

    # --- Tokenize Datasets ---
    print("\n--- Tokenizing Datasets ---")
    # Pass max_seq_length and model_type to preprocess_function
    tokenized_train = train_dataset.map(
        lambda examples: preprocess_function(examples, args.max_seq_length, args.model_type),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda examples: preprocess_function(examples, args.max_seq_length, args.model_type),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    print("Tokenized training dataset features:", tokenized_train)

    # --- Training Setup ---
    print("\n--- Setting Up Training ---")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args_dict = {
        "output_dir": model_output_path,
        "eval_strategy": "epoch",
        "learning_rate": effective_learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_epochs,
        "fp16": torch.cuda.is_available(),
        "predict_with_generate": True,
        "logging_dir": os.path.join(model_output_path, 'logs'),
        "logging_strategy": "steps",
        "disable_tqdm": False,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss", # or another metric like "eval_bleu" if preferred
        "greater_is_better": False, # True if metric_for_best_model is BLEU/METEOR etc.
        "generation_config": model.generation_config,
        "report_to": ["tensorboard"],
    }

    if args.push_to_hub and hub_model_id:
        training_args_dict["push_to_hub"] = True
        training_args_dict["hub_model_id"] = hub_model_id
        training_args_dict["hub_strategy"] = "end"
        # hub_token is handled by login() or TrainingArguments picks up env var / cached token
        if args.hf_token: # Explicitly pass if provided
             training_args_dict["hub_token"] = args.hf_token


    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    print("\nTraining Arguments Set.")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, # Uses global tokenizer set earlier
    )

    print("\nTrainer initialized. Starting training...")
    if torch.cuda.is_available():
        print(f"CUDA available. Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Training on CPU. This will be slow.")

    # --- Train ---
    try:
        train_result = trainer.train()
        print("\n--- Training Finished ---")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("Training metrics and state saved.")

        print("\n--- Evaluating Best Model on Validation Set ---")
        eval_metrics = trainer.evaluate(eval_dataset=tokenized_val)
        trainer.log_metrics("eval_final", eval_metrics)
        trainer.save_metrics("eval_final", eval_metrics)
        print("Final evaluation metrics:")
        print(json.dumps(eval_metrics, indent=2))

        print(f"\nBest model/adapter checkpoint saved locally within: {model_output_path}")

        # Trainer handles push_to_hub for the model/adapters if configured.
        # For LoRA, it pushes adapters. For full fine-tuning, it pushes the full model.
        # We might still want to push the tokenizer separately, especially for LoRA.
        if args.push_to_hub and hub_model_id:
            print(f"\nTrainer was configured to push to Hub ID: {hub_model_id}")
            print(f"Check repository status at: https://huggingface.co/{hub_model_id}")
            if args.use_lora: # For LoRA, also push the tokenizer for convenience
                try:
                    print(f"Attempting to push tokenizer to {hub_model_id}...")
                    tokenizer.push_to_hub(hub_model_id, token=args.hf_token if args.hf_token else None)
                    print("Tokenizer pushed successfully.")
                except Exception as e:
                    print(f"Error pushing tokenizer: {e}")
        else:
            print("\nPush to Hub was disabled or Hub ID not set. Model/adapters saved locally.")

    except Exception as e:
        print(f"\n--- An error occurred during training or evaluation ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n--- Harmonized Training Script Finished ---")

if __name__ == "__main__":
    main()