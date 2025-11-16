#!/usr/bin/env python3

import socket
import subprocess
import os
import time
import concurrent.futures
from pathlib import Path
import json

def get_free_port():
    """Get a free port for master port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_command(cmd, env=None):
    """Run a shell command with optional environment variables"""
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    else:
        full_env = None
    
    result = subprocess.run(cmd, shell=True, env=full_env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        return False
    return True

def training_completed(task_name):
    """Check if training is already completed by looking for model-00004-of-00004.safetensors"""
    model_file = Path(f"saves/unlearn/{task_name}/model-00004-of-00004.safetensors")
    if model_file.exists():
        print(f"Training already completed for {task_name} (found {model_file})")
        return True
    return False

def evaluation_completed(task_name, checkpoint):
    """Check if evaluation is already completed by looking for JSON file with 10 keys"""
    json_file = Path(f"saves/unlearn/{task_name}/checkpoint-{checkpoint}/evals/TOFU_SUMMARY.json")
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if len(data.keys()) >= 10:
                print(f"Evaluation already completed for {task_name}/checkpoint-{checkpoint} (found {len(data.keys())} keys)")
                return True
        except (json.JSONDecodeError, IOError):
            # If we can't read the file, assume it's incomplete
            pass
    return False

def get_files_to_delete(task_name, checkpoint=None):
    """Get list of files to delete for cleanup"""
    files_to_delete = []
    
    if checkpoint is None:
        # Top-level files to delete
        base_path = f"saves/unlearn/{task_name}"
        files_to_delete.extend([
            f"{base_path}/model-00001-of-00004.safetensors",
            f"{base_path}/model-00002-of-00004.safetensors",
            f"{base_path}/model-00003-of-00004.safetensors",
            f"{base_path}/model-00004-of-00004.safetensors",
            f"{base_path}/model.safetensors.index.json",
            f"{base_path}/config.json",
            f"{base_path}/generation_config.json",
            f"{base_path}/special_tokens_map.json",
            f"{base_path}/tokenizer_config.json",
            f"{base_path}/tokenizer.json",
            f"{base_path}/trainer_state.json",
            f"{base_path}/training_args.bin"
        ])
    else:
        # Checkpoint-specific files to delete
        checkpoint_path = f"saves/unlearn/{task_name}/checkpoint-{checkpoint}"
        files_to_delete.extend([
            f"{checkpoint_path}/model-00001-of-00004.safetensors",
            f"{checkpoint_path}/model-00002-of-00004.safetensors",
            f"{checkpoint_path}/model-00003-of-00004.safetensors",
            f"{checkpoint_path}/model-00004-of-00004.safetensors",
            f"{checkpoint_path}/model.safetensors.index.json",
            f"{checkpoint_path}/config.json",
            f"{checkpoint_path}/generation_config.json",
            f"{checkpoint_path}/special_tokens_map.json",
            f"{checkpoint_path}/tokenizer_config.json",
            f"{checkpoint_path}/tokenizer.json",
            f"{checkpoint_path}/trainer_state.json",
            f"{checkpoint_path}/training_args.bin"
        ])
    
    # Only return files that actually exist
    existing_files = []
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            existing_files.append(file_path)
    
    return existing_files

def delete_files(files_to_delete):
    """Delete the specified files"""
    if not files_to_delete:
        return
    
    print(f"Deleting {len(files_to_delete)} files to save space...")
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.system(f"rm '{file_path}'")
                print(f"  Deleted: {file_path}")
        except Exception as e:
            print(f"  Failed to delete {file_path}: {e}")

def run_eval_checkpoint(gpu_id, checkpoint, task_name, forget_split, holdout_split, model, retain_split):
    """Run evaluation for a single checkpoint"""
    # Check if evaluation is already completed
    if evaluation_completed(task_name, checkpoint):
        print(f"Evaluation already exists for checkpoint {checkpoint}, skipping execution")
        return checkpoint, True
    
    env = {'CUDA_VISIBLE_DEVICES': str(gpu_id)}
    
    cmd = f"""python src/eval.py \\
    experiment=eval/tofu/default.yaml \\
    forget_split={forget_split} \\
    holdout_split={holdout_split} \\
    model={model} \\
    task_name={task_name}/checkpoint-{checkpoint} \\
    model.model_args.pretrained_model_name_or_path=saves/unlearn/{task_name}/checkpoint-{checkpoint} \\
    paths.output_dir=saves/unlearn/{task_name}/checkpoint-{checkpoint}/evals \\
    retain_logs_path=saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json"""
    
    success = run_command(cmd, env)
    return checkpoint, success

def check_utility_zero(task_name, checkpoint):
    """
    Add your logic here to check if utility has gone to 0
    Return True if utility is 0 and we should stop, False otherwise
    """
    json_file = f"saves/unlearn/{task_name}/checkpoint-{checkpoint}/evals/TOFU_SUMMARY.json"
    try:
        with open(json_file, 'r') as f:
            info = json.load(f)
        
        if info.get("model_utility", 0) < 0.02:
            print(f"Utility reached 0 for {task_name} at checkpoint {checkpoint}")
            return True
        else:
            print(f"Utility not yet 0 for {task_name} at checkpoint {checkpoint} (utility={info.get('model_utility', 'N/A')})")
            return False
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or can't be read, assume utility hasn't reached 0
        return False

def main():
    # Set master port
    master_port = get_free_port()
    print(f"Master Port: {master_port}")
    
    # Configuration
    model = "Llama-3.1-8B-Instruct"
    
    trainer_experiments = {
        "GradAscent": "unlearn/tofu/default.yaml",
        "GradDiff": "unlearn/tofu/default.yaml",
        "NPO": "unlearn/tofu/default.yaml",
        "DPO": "unlearn/tofu/idk.yaml",
        "RMU": "unlearn/tofu/default.yaml"
    }
    
    epochs = 10  # Default epochs for all trainers
    
    # Split configuration
    split = "forget10 holdout10 retain90"
    split_parts = split.split()
    forget_split = split_parts[0]
    holdout_split = split_parts[1] 
    retain_split = split_parts[2]
    
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 1
    
    # Learning rates to test
    learning_rates = [2e-6, 1e-6, 3e-6, 1.5e-6, 4e-6, 8e-7, 5e-7]
    
    # Main training and evaluation loop
    for learning_rate in learning_rates:
        for trainer in trainer_experiments.keys():
            experiment = trainer_experiments[trainer]
            
            # Format learning rate for filename (e.g., 1e-5 -> 1e-5, 5e-6 -> 5e-6)
            lr_str = f"{learning_rate:.0e}".replace("+", "").replace("-0", "-")
            task_name = f"tofu_{model}_{forget_split}_{trainer}_{lr_str}"
            model_path = f"open-unlearning/tofu_{model}_full"
            
            print(f"{task_name}: Unlearning {model_path} using {trainer} at lr={learning_rate} for {epochs} epochs")
                
            # Check if training is already completed
            if training_completed(task_name):
                print(f"Skipping training for {task_name} - already completed")
            else:
                # Unlearn
                unlearn_env = {'CUDA_VISIBLE_DEVICES': '0,1'}
                unlearn_cmd = f"""accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port {master_port} \\
                src/train.py --config-name=unlearn.yaml \\
                experiment={experiment} \\
                trainer={trainer} \\
                task_name={task_name} \\
                model={model} \\
                forget_split={forget_split} \\
                retain_split={retain_split} \\
                model.model_args.pretrained_model_name_or_path={model_path} \\
                retain_logs_path=saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json \\
                trainer.args.per_device_train_batch_size={per_device_train_batch_size} \\
                trainer.args.gradient_accumulation_steps={gradient_accumulation_steps} \\
                trainer.args.num_train_epochs={epochs} \\
                trainer.args.learning_rate={learning_rate} \\
                trainer.args.ddp_find_unused_parameters=true \\
                trainer.args.gradient_checkpointing=true \\
                trainer.args.save_strategy=epoch"""
                
                if not run_command(unlearn_cmd, unlearn_env):
                    print(f"Unlearning failed for {task_name}")
                    continue
                
                # Delete top-level model files after training completes
                top_level_files = get_files_to_delete(task_name, checkpoint=None)
                delete_files(top_level_files)
            
            # Evaluation - Run checkpoints in pairs
            checkpoints = [13, 26, 39, 52, 65, 78, 91, 104, 117, 130]
            utility_zero_reached = False
            processed_checkpoints = []
            
            # Process checkpoints in pairs
            for i in range(0, len(checkpoints), 2):
                if utility_zero_reached:
                    # Delete remaining checkpoints since utility reached 0
                    remaining_checkpoints = checkpoints[i:]
                    for remaining_checkpoint in remaining_checkpoints:
                        remaining_files = get_files_to_delete(task_name, remaining_checkpoint)
                        delete_files(remaining_files)
                    break
                
                checkpoint1 = checkpoints[i]
                checkpoint2 = checkpoints[i+1] if i+1 < len(checkpoints) else None
                
                print(f"Evaluating checkpoints {checkpoint1}" + (f" and {checkpoint2}" if checkpoint2 else "") + " in parallel")
                
                # Run evaluations in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures = []
                    
                    # Submit first checkpoint evaluation
                    future1 = executor.submit(
                        run_eval_checkpoint, 
                        0, checkpoint1, task_name, forget_split, holdout_split, model, retain_split
                    )
                    futures.append((future1, checkpoint1))
                    
                    # Submit second checkpoint evaluation if it exists
                    if checkpoint2:
                        future2 = executor.submit(
                            run_eval_checkpoint,
                            1, checkpoint2, task_name, forget_split, holdout_split, model, retain_split
                        )
                        futures.append((future2, checkpoint2))
                    
                    # Wait for all evaluations to complete
                    for future, checkpoint in futures:
                        checkpoint_result, success = future.result()
                        if not success:
                            print(f"Evaluation failed for checkpoint {checkpoint_result}")
                        else:
                            processed_checkpoints.append(checkpoint_result)
                
                print(f"Completed evaluation of checkpoints {checkpoint1}" + (f" and {checkpoint2}" if checkpoint2 else ""))
                
                # Check utility for both checkpoints and clean up
                for checkpoint in [checkpoint1, checkpoint2]:
                    if checkpoint is None:
                        continue
                    
                    # Delete checkpoint files after evaluation
                    checkpoint_files = get_files_to_delete(task_name, checkpoint)
                    delete_files(checkpoint_files)
                    
                    # Check if utility reached 0
                    if check_utility_zero(task_name, checkpoint):
                        print(f"Utility has reached 0 at checkpoint {checkpoint}, will stop further evaluations for {task_name}")
                        utility_zero_reached = True
                        break
            
            # If utility reached 0, delete any remaining unprocessed checkpoints
            if utility_zero_reached:
                remaining_checkpoints = [cp for cp in checkpoints if cp not in processed_checkpoints]
                for remaining_checkpoint in remaining_checkpoints:
                    remaining_files = get_files_to_delete(task_name, remaining_checkpoint)
                    delete_files(remaining_files)

if __name__ == "__main__":
    main()