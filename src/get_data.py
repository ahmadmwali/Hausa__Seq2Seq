import re
import csv
import random
import os
import time
import math 
import argparse 

class TextProcessor:
    """
    A class to process text files: merge, clean, segment, apply noise,
    and save as separate TSV files for train, validation, and test sets,
    with configurable sizes.
    """
    HAUSA_CHARACTERS = "abcdefghijklmnopqrstuvwxyz'ƙɗɓyABCDEFGHIJKLMNOPQRSTUVWXYZ"

    DEFAULT_NOISE_LEVELS = {
        'random_spacing': 0.01,
        'remove_spaces': 0.1,
        'incorrect_characters': 0.01,
        'delete_characters': 0.01,
        'duplicate_characters': 0.01
    }

    def __init__(self, input_files, train_output_path, val_output_path, test_output_path,
                 train_size, val_size, test_size,
                 include_religious=False,
                 religious_files=None,
                 noise_levels=None, 
                 clean_output_path=None,
                 encoding='utf-8'):
        """
        Initializes the TextProcessor for train/val/test split output.
        """
        self.input_files = input_files
        self.train_output_path = train_output_path
        self.val_output_path = val_output_path
        self.test_output_path = test_output_path

        self.train_size = int(train_size) if train_size is not None and int(train_size) > 0 else None
        self.val_size = int(val_size) if val_size is not None and int(val_size) > 0 else None
        self.test_size = int(test_size) if test_size is not None and int(test_size) > 0 else None

        self.include_religious = include_religious
        self.religious_files = religious_files if religious_files is not None else ["Files/Bible.txt", "Files/Tanzil.txt"]
        
        self.noise_levels = noise_levels if noise_levels else self.DEFAULT_NOISE_LEVELS.copy()

        self.clean_output_path = clean_output_path
        self.encoding = encoding

        for key in self.DEFAULT_NOISE_LEVELS:
            if key not in self.noise_levels:
                print(f"Warning: Noise level for '{key}' not specified. Using default: {self.DEFAULT_NOISE_LEVELS[key]}")
                self.noise_levels[key] = self.DEFAULT_NOISE_LEVELS[key]
            elif not (0 <= self.noise_levels[key] <= 1):
                 raise ValueError(f"Noise probability for '{key}' must be between 0 and 1.")

        if self.train_size is None and self.val_size is None and self.test_size is None:
            raise ValueError("At least one of train_size, val_size, or test_size must be a positive integer or specified via arguments.")

    def _merge_files(self):
        """Merges content from specified input files."""
        print("Merging files...")
        merged_content = []
        files_to_process = self.input_files[:] 

        if not self.include_religious:
            print("Excluding religious files...")
            files_to_process = [
                fpath for fpath in files_to_process
                if not any(fpath.endswith(os.path.sep + r_file) or os.path.basename(fpath) == r_file for r_file in self.religious_files)
            ]
            print(f"Files remaining after exclusion: {[os.path.basename(f) for f in files_to_process]}")

        for fname in files_to_process:
            try:
                with open(fname, "r", encoding=self.encoding) as infile:
                    print(f"Reading: {fname}")
                    merged_content.append(infile.read())
            except FileNotFoundError:
                print(f"Warning: File not found - {fname}. Skipping.")
            except Exception as e:
                print(f"Warning: Could not read file {fname} due to {e}. Skipping.")

        print("File merging complete.")
        return "\n".join(merged_content) if merged_content else ""

    def _clean_text(self, text):
        """Cleans the text by removing unwanted patterns."""
        print("Cleaning text...")
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'NBSP', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        print("Text cleaning complete.")
        return text

    def _segment_sentences(self, text):
        """Splits text into sentences."""
        print("Segmenting text into sentences...")
        sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+')
        sentences = []
        paragraphs = text.split('\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                para_sentences = sentence_pattern.split(paragraph)
                sentences.extend([s.strip() for s in para_sentences if s.strip()])
        
        sentences = [s for s in sentences if s] # Filter out empty sentences
        print(f"Segmented into {len(sentences)} non-empty sentences.")
        return sentences

    def _random_spacing(self, text, probability):
        if probability == 0 or not text: return text
        new_text = []
        for i, char in enumerate(text):
            if char.isalnum() and random.random() < probability:
                 if i > 0 and text[i-1] != ' ':
                     new_text.append(' ')
            new_text.append(char)
        return re.sub(r' +', ' ', "".join(new_text)).strip()

    def _remove_spaces(self, text, probability):
        if probability == 0 or not text: return text
        new_text = list(text)
        for i in range(len(new_text) - 2, 0, -1):
            if new_text[i] == ' ':
                if new_text[i-1] != ' ' and new_text[i+1] != ' ':
                    if random.random() < probability:
                        new_text.pop(i)
        return "".join(new_text).strip()

    def _incorrect_characters(self, text, probability):
        if probability == 0 or not text: return text
        replacements = {'ɓ': 'b', 'ɗ': 'd', 'ƙ': 'k', 'ƴ': 'y',
                        'Ɓ': 'B', 'Ɗ': 'D', 'Ƙ': 'K', 'Ƴ': 'Y'}
        new_text = list(text)
        for i, char in enumerate(new_text):
             if char in replacements and random.random() < probability:
                 new_text[i] = replacements[char]
        return "".join(new_text)

    def _delete_characters(self, text, probability):
        if probability == 0 or not text: return text
        return ''.join(char for char in text if char == ' ' or random.random() > probability)

    def _duplicate_characters(self, text, probability):
        if probability == 0 or not text: return text
        new_text = []
        for char in text:
            new_text.append(char)
            if char != ' ' and random.random() < probability:
                new_text.append(char)
        return ''.join(new_text)

    def _apply_noise(self, text):
        if not text: return ""
        noisy_text = self._incorrect_characters(text, self.noise_levels.get('incorrect_characters', 0))
        noisy_text = self._delete_characters(noisy_text, self.noise_levels.get('delete_characters', 0))
        noisy_text = self._duplicate_characters(noisy_text, self.noise_levels.get('duplicate_characters', 0))
        noisy_text = self._random_spacing(noisy_text, self.noise_levels.get('random_spacing', 0))
        noisy_text = self._remove_spaces(noisy_text, self.noise_levels.get('remove_spaces', 0))
        noisy_text = re.sub(r'\s+', ' ', noisy_text).strip()
        return noisy_text

    def process_and_save(self):
        start_time = time.time()
        print("Starting text processing pipeline...")
        merged_text = self._merge_files()
        if not merged_text:
            print("Error: No text obtained after merging files. Aborting.")
            return

        cleaned_text = self._clean_text(merged_text)
        clean_sentences = self._segment_sentences(cleaned_text)
        total_sentences = len(clean_sentences)
        print(f"Total clean sentences generated: {total_sentences}")

        if total_sentences == 0:
            print("No sentences found after cleaning and segmentation. Aborting.")
            return

        if self.clean_output_path:
            print(f"Saving all clean sentences to {self.clean_output_path}...")
            try:
                clean_output_dir = os.path.dirname(self.clean_output_path)
                if clean_output_dir and not os.path.exists(clean_output_dir):
                    os.makedirs(clean_output_dir)
                with open(self.clean_output_path, 'w', encoding=self.encoding) as f_clean:
                    for sentence in clean_sentences:
                        f_clean.write(sentence + '\n')
                print(f"All {total_sentences} clean sentences saved.")
            except Exception as e:
                print(f"Error saving clean sentences file: {e}")

        print("Shuffling sentences...")
        random.shuffle(clean_sentences)

        current_idx = 0
        splits_data = []

        if self.train_size and self.train_size > 0 and current_idx < total_sentences:
            count = min(self.train_size, total_sentences - current_idx)
            splits_data.append(('Train', self.train_output_path, clean_sentences[current_idx : current_idx + count]))
            current_idx += count
        
        if self.val_size and self.val_size > 0 and current_idx < total_sentences:
            count = min(self.val_size, total_sentences - current_idx)
            splits_data.append(('Validation', self.val_output_path, clean_sentences[current_idx : current_idx + count]))
            current_idx += count

        if self.test_size and self.test_size > 0 and current_idx < total_sentences:
            count = min(self.test_size, total_sentences - current_idx)
            splits_data.append(('Test', self.test_output_path, clean_sentences[current_idx : current_idx + count]))
            current_idx += count
        
        print(f"Effective splits: Train={len(splits_data[0][2]) if len(splits_data)>0 else 0}, "
              f"Validation={len(splits_data[1][2]) if len(splits_data)>1 else 0}, "
              f"Test={len(splits_data[2][2]) if len(splits_data)>2 else 0}")


        for split_name, output_path, sentences_for_split in splits_data:
            if not sentences_for_split: # Skip if this split ended up empty
                print(f"Skipping {split_name} set as it has no sentences allocated.")
                continue

            print(f"\nProcessing and writing {split_name} set to {output_path} ({len(sentences_for_split)} sentences)...")
            rows_written = 0
            skipped_empty_noisy = 0

            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory for {split_name} output: {output_dir}")

            try:
                with open(output_path, 'w', encoding=self.encoding, newline='') as outfile:
                    writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['Noisy', 'Clean'])
                    for clean_sentence in sentences_for_split:
                         noisy_sentence = self._apply_noise(clean_sentence)
                         if noisy_sentence:
                             writer.writerow([noisy_sentence, clean_sentence])
                             rows_written += 1
                         else:
                             skipped_empty_noisy += 1
                print(f"Done writing {split_name} set. {rows_written} rows written.")
                if skipped_empty_noisy > 0:
                    print(f"Warning for {split_name} set: Skipped {skipped_empty_noisy} rows because the noisy version became empty.")
            except Exception as e:
                print(f"Error writing {split_name} TSV file: {e}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nProcessing pipeline finished in {duration:.2f} seconds.")

def parse_args():
    parser = argparse.ArgumentParser(description="Process text files to create noisy/clean datasets for training, validation, and testing.")
    
    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+', # Accepts one or more input files
        required=True,
        help="List of paths to the input text files."
    )
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="Files_Processed",
        help="Base directory to save the output TSV files and optional clean text file."
    )
    parser.add_argument(
        "--train_file_name", type=str, default="train.tsv", help="Name of the training output file (relative to output_dir_base)."
    )
    parser.add_argument(
        "--val_file_name", type=str, default="validation.tsv", help="Name of the validation output file (relative to output_dir_base)."
    )
    parser.add_argument(
        "--test_file_name", type=str, default="test.tsv", help="Name of the test output file (relative to output_dir_base)."
    )
    parser.add_argument(
        "--train_size", type=int, default=80000, help="Maximum number of samples for the training set. Set to 0 or omit to skip."
    )
    parser.add_argument(
        "--val_size", type=int, default=10000, help="Maximum number of samples for the validation set. Set to 0 or omit to skip."
    )
    parser.add_argument(
        "--test_size", type=int, default=10000, help="Maximum number of samples for the test set. Set to 0 or omit to skip."
    )
    parser.add_argument(
        "--include_religious",
        action="store_true", # Makes it a flag, default is False
        help="Include religious texts (as defined by --religious_files)."
    )
    parser.add_argument(
        "--religious_files",
        type=str,
        nargs='*', # Optional, can be multiple
        default=["Files/Bible.txt", "Files/Tanzil.txt"],
        help="List of filenames (can be paths) considered religious texts to be excluded if --include_religious is not set."
    )
    
    # Noise level arguments
    for noise_type, default_val in TextProcessor.DEFAULT_NOISE_LEVELS.items():
        parser.add_argument(
            f"--noise_{noise_type.replace('_', '-')}",
            type=float,
            default=default_val,
            help=f"Probability for {noise_type.replace('_', ' ')} noise. Default: {default_val}"
        )

    parser.add_argument(
        "--clean_output_file_name",
        type=str,
        default=None, 
        help="Name of the file to save all cleaned sentences. If not provided, clean sentences are not saved separately."
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding to use for reading and writing files."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and noise application for reproducibility."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # Construct noise_levels dictionary from args
    noise_levels_from_args = {}
    for noise_type in TextProcessor.DEFAULT_NOISE_LEVELS.keys():
        arg_name = f"noise_{noise_type.replace('_', '-')}"
        noise_levels_from_args[noise_type] = getattr(args, arg_name.replace('-', '_'))

    # Create output directory if it doesn't exist
    if args.output_dir_base and not os.path.exists(args.output_dir_base):
        os.makedirs(args.output_dir_base)
        print(f"Created main output directory: {args.output_dir_base}")

    # Construct full paths for output files
    train_path = os.path.join(args.output_dir_base, args.train_file_name) if args.train_size and args.train_size > 0 else None
    val_path = os.path.join(args.output_dir_base, args.val_file_name) if args.val_size and args.val_size > 0 else None
    test_path = os.path.join(args.output_dir_base, args.test_file_name) if args.test_size and args.test_size > 0 else None
    
    full_clean_output_path = None
    if args.clean_output_file_name:
        full_clean_output_path = os.path.join(args.output_dir_base, args.clean_output_file_name)

    # Instantiate and run the processor
    processor = TextProcessor(
        input_files=args.input_files,
        train_output_path=train_path,
        val_output_path=val_path,
        test_output_path=test_path,
        train_size=args.train_size if args.train_size > 0 else None, # Pass None if size is 0
        val_size=args.val_size if args.val_size > 0 else None,
        test_size=args.test_size if args.test_size > 0 else None,
        include_religious=args.include_religious,
        religious_files=args.religious_files,
        noise_levels=noise_levels_from_args,
        clean_output_path=full_clean_output_path,
        encoding=args.encoding
    )

    processor.process_and_save()

    # --- Example command to run this script ---
    # python Data_Clean.py \
    #   --input_files Files/Other_Texts.txt Files/Wikimedia.txt Files/Wikipedia.txt Files/Bible.txt Files/Tanzil.txt \
    #   --output_dir_base Processed_Data \
    #   --train_file_name train_data.tsv \
    #   --val_file_name val_data.tsv \
    #   --test_file_name test_data.tsv \
    #   --train_size 50000 \
    #   --val_size 5000 \
    #   --test_size 5000 \
    #   --noise_random_spacing 0.02 \
    #   --noise_remove_spaces 0.12 \
    #   --clean_output_file_name all_clean_sentences.txt \
    #   --seed 123
