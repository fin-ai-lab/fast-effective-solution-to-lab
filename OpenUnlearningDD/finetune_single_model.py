import sys
import pathlib
import os
import argparse
from typing import List, Tuple, Any, Optional
from pathlib import Path
import json
import re
import math

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers


def read_text(file_path: str) -> str:
    """Read text content from a .txt file."""
    if Path(file_path).suffix != '.txt':
        raise ValueError("File must have .txt extension")

    with open(file_path, 'r') as f:
        text: str = f.read()
    return text


def load_model(
    model_dir: str,
    model_name: Optional[str] = None,
    quantization_config: Any = None,
    reinforced_model_dir: Optional[str] = None
) -> AutoModelForCausalLM:
    """Load a language model with optional special configurations."""

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    return model


def load_tokenizer(
    tokenizer_dir: str,
    add_pad_token: bool = True,
    use_fast: bool = True
) -> AutoTokenizer:
    """Load a tokenizer with optional padding token configuration."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast) 
    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_dir: str,
    model_name: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    add_pad_token: bool = True,
    quantization_config: Any = None,
    reinforced_model_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load both model and tokenizer together."""
    model = load_model(
        model_dir, 
        model_name, 
        quantization_config,
        reinforced_model_dir=reinforced_model_dir
    )
    tokenizer = (
        load_tokenizer(tokenizer_dir, add_pad_token)
        if tokenizer_dir is not None
        else None
    )
    return model, tokenizer


def estimate_steps_per_epoch(
    samples: int,
    epochs: int,
    *_,
    per_device_batch_size: Optional[int] = None,
    batch_size: Optional[int] = None
) -> int:
    """Overestimate number of steps per epoch."""
    from torch.cuda import device_count
    from math import ceil

    if per_device_batch_size is None and batch_size is None:
        raise ValueError("Either per_device_batch_size or batch_size must be specified.")
    
    if batch_size is None:
        # per_device_batch_size is specified
        cnt = device_count()
        if cnt == 0:
            raise ValueError("Device not detected.")
        batch_size: int = device_count() * per_device_batch_size

    samples_per_epoch = ceil(samples / epochs)
    steps_per_epoch = ceil(samples_per_epoch / batch_size)
    return steps_per_epoch


def pad_or_trim_tensor(tensor: torch.Tensor, target_length: int, padding_value: int = 0) -> torch.Tensor:
    """Pad or trim tensor to target length."""
    current_length = tensor.size(0)
    
    if current_length < target_length:
        # Padding
        padding_size = target_length - current_length
        padding_tensor = torch.full((padding_size,), padding_value, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor
    elif current_length > target_length:
        # Trimming
        trimmed_tensor = tensor[:target_length]
        return trimmed_tensor
    else:
        # No change needed
        return tensor


class DefaultDataset(Dataset):
    """Dataset class for handling text data with tokenization."""

    def __init__(
        self,
        file_path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_len: Optional[int] = 4096,
        add_bos_token: bool = True
    ):
        self.input_ids = []
        self.strings = []
        
        if Path(file_path).suffix == '.json':
            self._load_json_data(file_path, tokenizer, max_len, add_bos_token)
        elif Path(file_path).suffix == '.txt':
            self._load_txt_data(file_path, tokenizer, max_len, add_bos_token)
        else:
            raise ValueError("File must be .json or .txt")

    def _load_json_data(self, file_path: str, tokenizer: AutoTokenizer, max_len: int, add_bos_token: bool):
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data[0], str):
            self.strings = data
        elif isinstance(data[0], dict) and 'text' in data[0] and isinstance(data[0]['text'], str):
            self.strings = [d['text'] for d in data]
            if 'input_ids' in data[0]:
                self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                return  # Done, since we have `input_ids` ready
        else:
            raise ValueError("Format of this `.json` file is not recognized.")

        assert tokenizer is not None, "Tokenizer must be specified."

        for s in self.strings:
            encoding: torch.Tensor = tokenizer(
                s,
                add_special_tokens=add_bos_token,
                return_tensors='pt'
            ).input_ids[0]
            encoding = pad_or_trim_tensor(
                encoding,
                target_length=max_len,
                padding_value=tokenizer.pad_token_id
            )
            self.input_ids.append(encoding)

    def _load_txt_data(self, file_path: str, tokenizer: AutoTokenizer, max_len: int, add_bos_token: bool):
        """Load data from text file."""
        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], 
                    (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.input_ids[index]

    def __len__(self) -> int:
        return len(self.input_ids)

    def get_collate_fn(self):
        """Return collate function for DataLoader."""
        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }
        return collate_fn


def finetune(
    model_dir: str,
    data_file: str,
    out_dir: str,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    epochs: int = 10,
    learning_rate: float = 1e-5,
    max_len: int = 2048,  # For unlearning they use 2048. For finetuning they use 4096. See Appendix B
    tokenizer_dir: Optional[str] = None
):
    """Fine-tune a language model on given data."""
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    model.gradient_checkpointing_enable()
    
    dataset = DefaultDataset(
        data_file,
        tokenizer=tokenizer,
        max_len=max_len
    )

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=5,
        save_strategy='no',
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        bf16=True,
        report_to='none'  # Disable wandb
    )
    
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn()
    )
    trainer.train()
    trainer.save_model(out_dir)


def determine_model_size(baseline_model_dir: str) -> str:
    """Determine model size based on baseline model directory."""
    if "1.3B" in baseline_model_dir:
        return "1.3b"
    elif "2.7B" in baseline_model_dir:
        return "2.7b"
    elif "7b" in baseline_model_dir:
        return "7b"
    else:
        # Default fallback - check if it's a path to an existing model
        if "1.3b" in baseline_model_dir:
            return "1.3b"
        elif "2.7b" in baseline_model_dir:
            return "2.7b"
        else:
            raise ValueError(f"Cannot determine model size from baseline: {baseline_model_dir}")


def get_model_config(baseline_model_dir: str) -> dict:
    """Get model configuration based on baseline model."""
    model_size = determine_model_size(baseline_model_dir)
    
    if model_size == "1.3b":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Using 1x GPU Config")
        return {
            'tokenizer_dir': "princeton-nlp/Sheared-LLaMA-1.3B",
            'learning_rate': 5e-5,
            'per_device_batch_size': 32,
            'gradient_accumulation_steps': 1 # 1x GPU
        }
    elif model_size == "2.7b":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Using 1x GPU Config")
        return {
            'tokenizer_dir': "princeton-nlp/Sheared-LLaMA-2.7B",
            'learning_rate': 4e-5,
            'per_device_batch_size': 32,
            'gradient_accumulation_steps': 1 # 1x GPU
        }
    elif model_size == "7b":
        print("Using 2x GPU Config")
        return {
            'tokenizer_dir': "meta-llama/Llama-2-7b-hf",
            'learning_rate': 2.5e-5,
            'per_device_batch_size': 16,
            'gradient_accumulation_steps': 1 # 2x GPU
        }
    else:
        raise ValueError(f"Unsupported model size: {model_size}")


def finetune_model(save_directory: str, data_file: str, baseline_model_dir: str):
    """Fine-tune a model with size-appropriate configurations."""
    
    if os.path.exists(save_directory):
        print(f"Directory {save_directory} already exists, skipping...")
        return
    
    config = get_model_config(baseline_model_dir)
    
    print(f"Using learning rate: {config['learning_rate']}")
    print(f"Using batch size: {config['per_device_batch_size']} with gradient accumulation steps: {config['gradient_accumulation_steps']}")
    print(f"Using tokenizer: {config['tokenizer_dir']}")
        
    finetune(
        model_dir=baseline_model_dir,
        data_file=data_file,
        out_dir=save_directory,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        tokenizer_dir=config['tokenizer_dir'],
        learning_rate=config['learning_rate'],
        epochs=10,
        per_device_batch_size=config['per_device_batch_size'],
        max_len=2048
    )


def main():
    """Main function to handle command line arguments and run training."""
    parser = argparse.ArgumentParser(description='Finetune a single model')
    parser.add_argument('model_number', type=int, help='Model number')
    parser.add_argument('data_file', type=str, help='Data file path')
    parser.add_argument('baseline_model_dir', type=str, help='Baseline model directory to finetune from')
   
    args = parser.parse_args()
    
    # Determine model size and create appropriate directory structure
    model_size = determine_model_size(args.baseline_model_dir)
    model_directory = f"models/{model_size}/"
    save_directory = model_directory + f"model_{args.model_number}"
   
    print(f"Training model {args.model_number} with data file: {args.data_file}")
    print(f"Using baseline model: {args.baseline_model_dir}")
    print(f"Model size: {model_size}")
    print(f"Save directory: {save_directory}")
   
    finetune_model(save_directory, args.data_file, args.baseline_model_dir)
   
    print(f"Completed training model {args.model_number}")


if __name__ == "__main__":
    main()