#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import warnings
import argparse
from tqdm.auto import tqdm
import traceback 

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
from huggingface_hub import login

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for a column in a DataFrame using a Seq2Seq model.")
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
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file (TSV or CSV)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output data file (TSV or CSV) with predictions."
    )
    parser.add_argument(
        "--input_text_column",
        type=str,
        required=True,
        help="Name of the column in the input file containing the text to make predictions on."
    )
    parser.add_argument(
        "--prediction_column_name",
        type=str,
        default="generated_prediction",
        help="Name for the new column where predictions will be stored."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for private models/repos."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Max sequence length for tokenizer input."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for generation."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search generation."
    )
    parser.add_argument(
        "--prediction_batch_size",
        type=int,
        default=8,
        help="Batch size for generating predictions."
    )
    # BART/M2M100 specific tokenizer params
    parser.add_argument(
        "--source_lang_code",
        type=str,
        default="ha", # Default for Hausa
        help="Source language code for mBART/M2M100 tokenizers (e.g., 'ha', 'en_XX'). Only used if model_type is 'bart'."
    )
    parser.add_argument(
        "--target_lang_code",
        type=str,
        default="ha", # Default for Hausa
        help="Target language code for mBART/M2M100 tokenizers (e.g., 'ha', 'en_XX'). Only used if model_type is 'bart'."
    )
    parser.add_argument(
        "--mount_drive",
        action="store_true",
        help="Mount Google Drive (for Colab)."
    )
    parser.add_argument(
        "--file_separator",
        type=str,
        default="\t", # Default to tsv.
        help="Single-character separator for input/output files (e.g., '\\t' for TSV, ',' for CSV). "
             "If an invalid separator is provided, the script will attempt to infer based on file extension."
    )


    args = parser.parse_args()

    if args.is_lora_adapter and not PEFT_AVAILABLE:
        raise ImportError("PEFT library is not installed, but --is_lora_adapter was specified. Please install peft: pip install peft")
    if args.is_lora_adapter and not args.base_model_name_for_lora:
        print("Warning: --is_lora_adapter is True, but --base_model_name_for_lora is not set. "
              "Attempting to infer base model from PeftConfig. If this fails, please specify it.")
    return args

def load_model_and_tokenizer(args):
    """Loads the model and tokenizer based on provided arguments."""
    print(f"Loading model: {args.hub_model_id}, Type: {args.model_type.upper()}, LoRA: {args.is_lora_adapter}")

    tokenizer_load_args = {"token": args.hf_token}
    model_load_args = {"token": args.hf_token}
    base_model_name_resolved = args.base_model_name_for_lora

    if args.model_type == "bart":
        tokenizer_load_args['src_lang'] = args.source_lang_code
        tokenizer_load_args['tgt_lang'] = args.target_lang_code
        print(f"BART model: src_lang='{args.source_lang_code}', tgt_lang='{args.target_lang_code}'")

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

    else: # Full model
        tokenizer = AutoTokenizer.from_pretrained(args.hub_model_id, **tokenizer_load_args)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.hub_model_id, **model_load_args)
        print("Full model loaded successfully.")

    model.eval() # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded onto device: {device}")

    # Setup Generation Configuration
    if hasattr(model, 'generation_config'):
        generation_config = model.generation_config
    else: # Should exist on base model or full model
        generation_config = GenerationConfig.from_model_config(model.config)

    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.num_beams = args.num_beams
    # generation_config.early_stopping = True

    if args.model_type == "bart": # Special handling for mBART/M2M100 forced BOS token
        try:
            if hasattr(tokenizer, 'get_lang_id'): # For M2M100Tokenizer, NllbTokenizer
                forced_bos_id = tokenizer.get_lang_id(args.target_lang_code)
            elif hasattr(tokenizer, 'lang_code_to_id') and args.target_lang_code in tokenizer.lang_code_to_id: # For MBart50Tokenizer
                forced_bos_id = tokenizer.lang_code_to_id[args.target_lang_code]
            else: # Fallback or if not applicable
                forced_bos_id = None
                is_multilingual_bart_variant = False
                if args.is_lora_adapter and args.base_model_name_for_lora:
                    model_id_check = args.base_model_name_for_lora.lower()
                    if "m2m100" in model_id_check or "mbart" in model_id_check:
                        is_multilingual_bart_variant = True
                elif not args.is_lora_adapter:
                    model_id_check = args.hub_model_id.lower()
                    if "m2m100" in model_id_check or "mbart" in model_id_check:
                        is_multilingual_bart_variant = True
                
                if is_multilingual_bart_variant:
                     print(f"Warning: Could not determine forced_bos_token_id method for lang '{args.target_lang_code}' with BART model type. This might be critical for multilingual models.")

            if forced_bos_id is not None:
                generation_config.forced_bos_token_id = forced_bos_id
                print(f"Set forced_bos_token_id: {forced_bos_id} for target language '{args.target_lang_code}'")
        except Exception as e_gc:
            print(f"Warning: Could not set forced_bos_token_id for BART model: {e_gc}")
    
    model.generation_config = generation_config # Attach updated config to model
    print(f"Generation config: max_new_tokens={model.generation_config.max_new_tokens}, num_beams={model.generation_config.num_beams}")
    if hasattr(model.generation_config, 'forced_bos_token_id'):
        print(f"  forced_bos_token_id: {model.generation_config.forced_bos_token_id}")

    return model, tokenizer, device

def generate_predictions_batch(texts, model, tokenizer, device, args):
    """Generates predictions for a batch of texts."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_seq_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=model.generation_config # Use the config attached to the model
        )
    
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [pred.strip() for pred in decoded_preds]

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

    if args.hf_token:
        try:
            login(token=args.hf_token)
            print("Logged in to Hugging Face Hub using provided token.")
        except Exception as e:
            print(f"Hugging Face login failed: {e}. Proceeding without explicit login.")

    # --- Load Model and Tokenizer ---
    try:
        model, tokenizer, device = load_model_and_tokenizer(args)
    except Exception as e:
        print(f"Fatal error during model/tokenizer loading: {e}")
        traceback.print_exc()
        return # Exit if model loading fails

    # --- Load DataFrame ---
    print(f"\n--- Loading DataFrame from: {args.input_file} ---")
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found at {args.input_file}")
        return
    
    try:
        # Determine separator for input file
        final_input_sep = args.file_separator

        if not isinstance(final_input_sep, str) or len(final_input_sep) != 1:
            print(f"Warning: --file_separator ('{args.file_separator}') is not a single character. Attempting to infer from input file extension.")
            if args.input_file.lower().endswith(".csv"):
                final_input_sep = ","
                print(f"Inferred input separator as ',' for .csv file.")
            elif args.input_file.lower().endswith(".tsv"):
                final_input_sep = "\t"
                print(f"Inferred input separator as '\\t' for .tsv file.")
            else:
                final_input_sep = "\t" # Default fallback if not a single char and no known extension
                print(f"Could not infer input separator from file extension '{args.input_file}'. Defaulting to tab ('\\t').")
        
        # Final check, should not be needed if logic above is correct, but as a safeguard.
        if not (isinstance(final_input_sep, str) and len(final_input_sep) == 1):
             print(f"Critical Warning: Input separator ('{final_input_sep}') is still invalid after checks. Forcing to tab ('\\t').")
             final_input_sep = "\t"
        
        df = pd.read_csv(args.input_file, sep=final_input_sep, on_bad_lines='warn')
        print(f"Loaded DataFrame with {len(df)} rows using separator: {repr(final_input_sep)}.")
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        traceback.print_exc()
        return

    if args.input_text_column not in df.columns:
        print(f"ERROR: Input text column '{args.input_text_column}' not found in the DataFrame.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # --- Generate Predictions ---
    print(f"\n--- Generating predictions for column: '{args.input_text_column}' ---")
    all_predictions = []
    input_texts = df[args.input_text_column].astype(str).tolist() # Ensure all inputs are strings

    for i in tqdm(range(0, len(input_texts), args.prediction_batch_size), desc="Predicting Batches"):
        batch_texts = input_texts[i:i + args.prediction_batch_size]
        try:
            batch_predictions = generate_predictions_batch(batch_texts, model, tokenizer, device, args)
            all_predictions.extend(batch_predictions)
        except Exception as e:
            print(f"Error during prediction for batch starting at index {i}: {e}")
            traceback.print_exc() # Print traceback for prediction errors
            # Add placeholders for failed batch predictions to maintain DataFrame alignment
            all_predictions.extend(["ERROR_GENERATING_PREDICTION"] * len(batch_texts))

    if len(all_predictions) != len(df):
        print(f"Warning: Number of predictions ({len(all_predictions)}) does not match number of rows in DataFrame ({len(df)}). This may cause issues.")
        # Pad predictions if necessary
        if len(all_predictions) < len(df):
            all_predictions.extend(["MISSING_PREDICTION"] * (len(df) - len(all_predictions)))
        else: # Truncate if too many
            all_predictions = all_predictions[:len(df)]

    df[args.prediction_column_name] = all_predictions
    print(f"Predictions added to new column: '{args.prediction_column_name}'")

    # --- Save DataFrame ---
    print(f"\n--- Saving DataFrame with predictions to: {args.output_file} ---")
    try:
        # Determine separator for output file
        final_output_sep = args.file_separator

        if not isinstance(final_output_sep, str) or len(final_output_sep) != 1:
            print(f"Warning: --file_separator ('{args.file_separator}') is not a single character. Attempting to infer from output file extension.")
            if args.output_file.lower().endswith(".csv"):
                final_output_sep = ","
                print(f"Inferred output separator as ',' for .csv file.")
            elif args.output_file.lower().endswith(".tsv"):
                final_output_sep = "\t"
                print(f"Inferred output separator as '\\t' for .tsv file.")
            else:
                final_output_sep = "\t" # Default fallback
                print(f"Could not infer output separator from file extension '{args.output_file}'. Defaulting to tab ('\\t').")

        if not (isinstance(final_output_sep, str) and len(final_output_sep) == 1):
             print(f"Critical Warning: Output separator ('{final_output_sep}') is still invalid after checks. Forcing to tab ('\\t').")
             final_output_sep = "\t"

        df.to_csv(args.output_file, sep=final_output_sep, index=False)
        print(f"DataFrame saved successfully using separator: {repr(final_output_sep)}.")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")
        traceback.print_exc()

    print("\n--- Prediction Script Finished ---")

if __name__ == "__main__":
    main()
