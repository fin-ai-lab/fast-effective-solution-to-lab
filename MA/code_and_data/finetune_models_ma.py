# scp finetune_models_ma.py ubuntu@192.222.52.171:~/
# scp data/august_19/ma_data.csv ubuntu@192.222.52.171:~/
# scp data/august_29/* ubuntu@192.222.52.171:~/
# python -m pip install -U transformers accelerate bitsandbytes 'jinja2>=3.1.0' tf-keras dotenv tabulate seaborn
# huggingface-cli login
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import random
import json
import torch
import os
from dotenv import load_dotenv

load_dotenv()
CONFIG = os.getenv("CONFIG", "CLUSTER")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if CONFIG == "LOCAL":
    split_a_file = 'data/august_19/split_a.txt'
    split_b_file = 'data/august_19/split_b.txt'
    distill_results_path = "data/august_19/distill_output/distill_results.jsonl"
    save_path = "../../../../data/lab/ma_finetuned_models/"
else:
    distill_results_path = "distill_results.jsonl"
    save_path = ""
    split_a_file = 'split_a.txt'
    split_b_file = 'split_b.txt'


class MADistillationDataset(Dataset):
    def __init__(self, 
                 tokenizer,
                 split_file: str,
                 validation_split: bool = False,
                 final_training: bool = False,
                 seed: int = 42):
        """
        Dataset that uses distillation responses from the 27b model as training data
        
        Args:
            tokenizer: HuggingFace tokenizer
            split_file: Path to file containing deal numbers for this split
            validation_split: If True, creates validation set, otherwise training set
            final_training: If True, uses all data regardless of validation_split
            seed: Random seed for sampling
        """
        self.tokenizer = tokenizer
        self.validation_split = validation_split
        self.final_training = final_training
        
        # Load split deal numbers
        with open(split_file, "r") as f:
            split_deals = set(f.read().splitlines())
        split_deals = {int(deal.strip()) for deal in split_deals if deal.strip()}
        
        # For each deal, randomly select a prompt_id for validation (1-8)
        random.seed(seed)
        deal_validation_prompts = {}
        for deal in split_deals:
            deal_validation_prompts[deal] = str(random.randint(1, 8))
        
        # Load distillation results
        self.training_examples = []
        
        if os.path.exists(distill_results_path):
            print(f"Loading distillation results from {distill_results_path}")
            
            with open(distill_results_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        deal_no = record.get('deal_no')
                        prompt_id = record.get('prompt_id', '')
                        
                        # Only include deals in our split
                        if deal_no not in split_deals:
                            continue
                            
                        # Skip if no response or error
                        response = record.get('response', '')
                        if not response:
                            continue
                        
                        # If final training, include everything
                        if self.final_training:
                            self.training_examples.append(response)
                        else:
                            # Determine if this record should be in validation or training
                            is_validation_record = (prompt_id == deal_validation_prompts[deal_no])
                            
                            # Include this record based on what split we're creating
                            if self.validation_split == is_validation_record:
                                self.training_examples.append(response)
                        
                    except json.JSONDecodeError:
                        continue
        
        if self.final_training:
            split_type = "final training (all data)"
        else:
            split_type = "validation" if validation_split else "training"
        print(f"Created {split_type} dataset with {len(self.training_examples)} examples from {len(split_deals)} deals")
        
        # Shuffle the final training examples
        random.seed(seed + (1 if validation_split else 0))  # Different seed for val/train
        random.shuffle(self.training_examples)
    
    def __len__(self):
        return len(self.training_examples)
    
    def __getitem__(self, idx):
        """Returns a training example formatted for the 4b model - FIXED VERSION"""
        input_text = self.training_examples[idx]
        
        # Method 1: Find the exact token boundary using the full tokenization
        # First tokenize the complete text
        full_encoding = self.tokenizer(
            input_text,
            padding=False,  # Don't pad yet
            truncation=False,  # Don't truncate yet
            return_tensors=None,
            add_special_tokens=False  # Your data already has special tokens
        )
        
        full_input_ids = full_encoding['input_ids']
        
        # Find where "<start_of_turn>model\n" appears in the token sequence
        model_marker = "<start_of_turn>model\n"
        
        # Strategy: Find this marker by converting tokens back to text
        mask_until_idx = 0
        for i in range(1, len(full_input_ids) + 1):
            partial_text = self.tokenizer.decode(full_input_ids[:i], skip_special_tokens=False)
            if model_marker in partial_text:
                # Find exactly where the marker ends
                marker_end_pos = partial_text.find(model_marker) + len(model_marker)
                
                # Now find which token index corresponds to the end of the marker
                for j in range(i, len(full_input_ids) + 1):
                    test_text = self.tokenizer.decode(full_input_ids[:j], skip_special_tokens=False)
                    if len(test_text) >= marker_end_pos:
                        mask_until_idx = j
                        break
                break
        
        # Now apply padding and truncation
        if len(full_input_ids) > 1600:
            # If too long, prioritize keeping the model response
            if mask_until_idx < 1600:
                # Keep all of user part and truncate model response
                full_input_ids = full_input_ids[:1600]
            else:
                # User part is too long, this is problematic
                print(f"Warning: User prompt too long ({mask_until_idx} tokens), truncating...")
                full_input_ids = full_input_ids[:1600]
                mask_until_idx = min(mask_until_idx, 1600)
        
        # Pad to max length
        if len(full_input_ids) < 1600:
            padding_length = 1600 - len(full_input_ids)
            full_input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = [1] * (1600 - padding_length) + [0] * padding_length
        else:
            attention_mask = [1] * 1600
        
        # Create labels with proper masking
        labels = full_input_ids.copy()
        
        # Mask everything up to (and including) the model marker
        for i in range(min(mask_until_idx, len(labels))):
            labels[i] = -100
        
        # Also mask padding tokens
        for i in range(len(labels)):
            if attention_mask[i] == 0:  # Padding token
                labels[i] = -100
        
        return {
            'input_ids': full_input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def finetune_distillation(
    model_dir: str,
    mini_batch_size,
    split_file: str,
    out_dir: str,
    epochs: int = 2,  
    learning_rate: float = 1e-5,
    final: bool = False,
):
    """
    Finetune a small model using distilled responses from a larger model
    
    Args:
        final: If True, trains on all data without validation split or checkpoints
    """
    if out_dir == model_dir:
        raise ValueError("out_dir and model_dir cannot be the same")

    print(f"Starting distillation finetuning:")
    print(f"  Model: {model_dir}")
    print(f"  Split file: {split_file}")
    print(f"  Output: {out_dir}")
    print(f"  Final training: {final}")

    os.system(f"rm -rf {out_dir}/")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    if final:
        # For final training, use all data
        train_dataset = MADistillationDataset(
            tokenizer, 
            split_file,
            final_training=True,
        )
        val_dataset = None
        
        if len(train_dataset) == 0:
            raise ValueError(f"No training examples found! Check that distillation results exist and deals match split file.")
        
        print(f"Final training dataset size: {len(train_dataset)}")
        
    else:
        # Create training dataset
        train_dataset = MADistillationDataset(
            tokenizer, 
            split_file,
            validation_split=False,  # Training set
        )
        
        # Create validation dataset
        val_dataset = MADistillationDataset(
            tokenizer, 
            split_file,
            validation_split=True,   # Validation set
        )
        
        if len(train_dataset) == 0:
            raise ValueError(f"No training examples found! Check that distillation results exist and deals match split file.")
        
        if len(val_dataset) == 0:
            raise ValueError(f"No validation examples found! Check that distillation results exist and deals match split file.")
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Validation ratio: {len(val_dataset) / len(train_dataset):.3f}")

    # Data collator
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    gradient_accumulation_steps = 32 // mini_batch_size

    if final:
        # Final training arguments - no validation, no checkpoints
        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=mini_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            save_strategy='no',  # Don't save checkpoints during training
            eval_strategy='no',  # No evaluation during training
            num_train_epochs=epochs,
            optim='adamw_torch',
            lr_scheduler_type='cosine',
            bf16=True,
            report_to='none',
            dataloader_drop_last=False,
            logging_steps=1,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
    else:
        # Regular training arguments with validation
        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=mini_batch_size,
            per_device_eval_batch_size=mini_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            save_strategy='epoch',  # Changed to save at each epoch
            eval_strategy='epoch',  # Evaluate at each epoch
            num_train_epochs=epochs,
            optim='adamw_torch',
            lr_scheduler_type='cosine',
            bf16=True,
            report_to='none',
            dataloader_drop_last=False,
            logging_steps=1,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            load_best_model_at_end=True,  # Load best model based on validation loss
            metric_for_best_model='eval_loss',
            greater_is_better=False,  # Lower loss is better
            # save_total_limit=2,  # Keep only 2 best checkpoints
        )
    
    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # None for final training
        args=training_args,
        data_collator=data_collator,
    )
    
    if not final:
        print("First eval...")
        print(trainer.evaluate())
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final model to {out_dir}")
    trainer.save_model(out_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    print("Starting distillation-based finetuning...")
    
    # Train model B on split B
    print("\n" + "="*50)
    print("Training Model B on Split B")
    print("="*50)
    finetune_distillation(
        model_dir='google/gemma-3-4b-it',
        mini_batch_size=4,
        epochs=3,
        split_file=split_b_file,
        out_dir=f'{save_path}4b_b/',
        final=True,
    )

    # Train model A on split A
    print("\n" + "="*50)
    print("Training Model A on Split A")
    print("="*50)
    finetune_distillation(
        model_dir='google/gemma-3-4b-it',
        mini_batch_size=4,
        epochs=3,
        split_file=split_a_file,
        out_dir=f'{save_path}4b_a/',
        final=True,
    )