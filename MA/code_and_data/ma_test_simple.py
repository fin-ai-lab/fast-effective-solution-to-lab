import re
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from transformers import BitsAndBytesConfig
import multiprocessing as mp
import os
import glob
import fcntl
import time
import queue
import threading
import seaborn as sns
import matplotlib.pyplot as plt
from gemma_dd import GemmaItDD
from dotenv import load_dotenv
import hashlib

# Load environment config and set all paths at startup
load_dotenv()
CONFIG = os.getenv("CONFIG", "CLUSTER")
print(f"Running with config: {CONFIG}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'figure.titleweight': 'bold'
})

# ============================================================================
# CONFIGURATION - All paths and directories set here
# ============================================================================
if CONFIG == "LOCAL":
    CSV_PATH = "data/august_19/ma_data.csv"
    DISTILL_DEALS_PATH = "data/august_19/distill_output/distill_deals_v2.txt"
    SPLIT_A_PATH = os.path.join("data/august_29/", "split_a.txt")
    SPLIT_B_PATH = os.path.join("data/august_29/", "split_b.txt")
    DEFAULT_OUTPUT_FOLDER = "data/august_19/"
    DEFAULT_RESULTS_FOLDER = "data/august_29/lab"
    DEFAULT_PLOT_OUTPUT = "data/august_29/mention_rates_plot"
else:
    # Cluster/production paths
    DATA_DIR = ""
    CSV_PATH = "ma_data.csv"
    DISTILL_DEALS_PATH = "distill_deals_v2.txt"
    SPLIT_A_PATH = "split_a.txt"
    SPLIT_B_PATH = "split_b.txt"
    DEFAULT_OUTPUT_FOLDER = ""
    DEFAULT_RESULTS_FOLDER = ""
    DEFAULT_PLOT_OUTPUT = "mention_rates_plot"
    MODEL_PATH_4B_A = "4b_a/"
    MODEL_PATH_4B_B = "4b_b/"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mention_detected(text, target_name):
    suffixes_to_remove = [
        'inc', 'corp', 'corporation', 'ltd', 'co', 'llc', 'group', 'plc', 'intl', 'incorporated',
        'holdings', 'limited', 'sa', 'nv', 'bv', 'ag', 'kg', 'gmbh', 'sarl', 'holding', 'international'
    ]
    target_name = target_name.lower()
    if "\"" in target_name:
        target_name = target_name.replace("\"", "")
    for suffix in suffixes_to_remove:
        target_name = target_name.replace(suffix, '').strip()
   
    option2 = target_name
    if "-" in target_name:
        option2 = option2.replace("-", " ")
   
    # Use word boundaries to match complete words only
    def has_word_match(text_lower, pattern):
        return bool(re.search(r'\b' + re.escape(pattern) + r'\b', text_lower))
   
    text_lower = text.lower()
    return (has_word_match(text_lower, target_name) or
            (option2 != target_name and has_word_match(text_lower, option2)))

def safe_write_to_file(filepath, record, max_retries=5):
    """Write a record to file with file locking and retry logic"""
    for attempt in range(max_retries):
        try:
            with open(filepath, "a") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    return True
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                continue
            else:
                print(f"Failed to write to {filepath} after {max_retries} attempts: {e}")
                return False
    return False

def load_existing_keys(filepath):
    """Load existing keys from the file with error handling"""
    existing_keys = set()
    if not os.path.exists(filepath):
        return existing_keys
    
    try:
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'key' in data:
                        existing_keys.add(data['key'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {line_num} in {filepath}")
                    continue
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
    
    return existing_keys

def load_model_and_tokenizer(model_name, device="cuda:0"):
    """Load the model and tokenizer"""
    if "google/" in model_name:
        print(f"Loading model {model_name} on {device}")
        
        device_map = {"": device}
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    else:
        print(f"Loading custom model {model_name} on {device}")
        
        if "b" in model_name:  # unlearning set b
            model = GemmaItDD(MODEL_PATH_4B_A, MODEL_PATH_4B_B, device=device)
        elif "a" in model_name:  # unlearning set a
            model = GemmaItDD(MODEL_PATH_4B_B, MODEL_PATH_4B_A, device=device)
        tokenizer = model.tokenizer

        return model, tokenizer

def generate_response(model, tokenizer, acquirer_name, year, alpha=None, topk=None, seed=None):
    """Generate a response for a given M&A transaction"""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    #I think the second prompt is about what I used the first time I did this though I couldn't find it in my git history...
    #        {"role": "user", "content": f"What firms did {acquirer_name} acquire in {year}? List without explanation."},
    messages = [
        {"role": "user", "content": f"It's the end of {year-1}. What two or three companies do you think {acquirer_name} might most consider acquiring in {year}?"},
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    if alpha is None and topk is None:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.4,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        # For GemmaItDD, use the custom generate method
        response = model.generate(inputs['input_ids'], alpha=alpha, topk=topk, temperature=0.4, seed=seed)
    
    return response

def key_to_seed(key):
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 ** (32 - 1)
    return seed

def key_to_components(key):
    """Extract components from key: master_deal_no_model_name_run_X"""
    parts = key.split('_run_')
    if len(parts) != 2:
        return None, None, None
    
    run = int(parts[1])
    remaining = parts[0]
    
    # Find model name by looking for known model patterns
    model_start = remaining.find('_')
    if model_start == -1:
        return None, None, None
        
    master_deal_no = int(remaining[:model_start])
    model_from_key = remaining[model_start+1:]
    
    return master_deal_no, model_from_key, run

def progress_monitor(work_queue, total_keys, update_interval=30):
    """Monitor and report progress periodically"""
    while True:
        try:
            remaining = work_queue.qsize()
            completed = total_keys - remaining
            if total_keys > 0:
                progress = (completed / total_keys) * 100
                print(f"Progress: {completed}/{total_keys} ({progress:.1f}%) - {remaining} keys remaining")
            
            if remaining == 0:
                break
                
            time.sleep(update_interval)
        except:
            break

def dynamic_worker(work_queue, progress_queue, output_file, model_name, device="cuda:0", timeout=60):
    """Dynamic worker that pulls work from shared queue"""
    print(f"Worker on {device}: Starting up...")
    
    try:
        torch.cuda.set_device(device)
        if "dd" in model_name:
            alpha = int(model_name[-1])
            topk = None
        elif "dk" in model_name:
            topk = int(model_name.split("_")[-1])
            alpha = None
        else:
            topk = None
            alpha = None

        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # Load data once
        data = pd.read_csv(CSV_PATH)
        with open(DISTILL_DEALS_PATH, 'r') as f:
            distill_deals = f.read().splitlines()
        distill_deals = [int(d.strip()) for d in distill_deals if d.strip()]
        data = data[data['master_deal_no'].isin(distill_deals)]
        data['dateann'] = pd.to_datetime(data['dateann'])
        data['year'] = data['dateann'].dt.year
        data = data.reset_index(drop=True)
        
        # Create a lookup dict for faster access
        data_lookup = {row['master_deal_no']: row for _, row in data.iterrows()}
        
        print(f"Worker on {device}: Model loaded, starting work...")
        
        processed_count = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3  # Allow 3 timeouts before giving up
        
        while True:
            try:
                # Get next key from queue with timeout
                unique_key = work_queue.get(timeout=timeout)
                consecutive_timeouts = 0  # Reset timeout counter on successful get
                
                # Check for poison pill (shutdown signal)
                if unique_key is None:
                    print(f"Worker on {device}: Received shutdown signal")
                    break
                
                # Process the key
                master_deal_no, _, run = key_to_components(unique_key)
                
                if master_deal_no is None or master_deal_no not in data_lookup:
                    print(f"Worker on {device}: Skipping invalid key {unique_key}")
                    work_queue.task_done()
                    continue
                    
                obs = data_lookup[master_deal_no]
                acquirer_name = obs['amanames']
                target_name = obs['tmanames']
                year = obs['year']
                
                try:
                    response = generate_response(
                        model, tokenizer, acquirer_name, year, 
                        seed=key_to_seed(unique_key),
                        alpha=alpha,
                        topk=topk
                    )
                    
                    mention_found = mention_detected(response, target_name)
                    
                    record = {
                        "key": unique_key,
                        "master_deal_no": master_deal_no,
                        "acquirer_name": acquirer_name,
                        "target_name": target_name,
                        "year": year,
                        "response": response,
                        "mention_detected": mention_found,
                        "run": run,
                        "device": device
                    }
                    
                    # Use safe file writing with locking
                    if safe_write_to_file(output_file, record):
                        processed_count += 1
                        
                        # Report progress
                        progress_queue.put(('progress', device, processed_count))
                    else:
                        print(f"Worker on {device}: Failed to save record for {unique_key}")
                        
                except Exception as e:
                    print(f"Worker on {device}: Error processing {unique_key}: {str(e)}")
                    record = {
                        "key": unique_key,
                        "master_deal_no": master_deal_no,
                        "acquirer_name": acquirer_name,
                        "target_name": target_name,
                        "year": year,
                        "response": "",
                        "mention_detected": False,
                        "run": run,
                        "error": str(e),
                        "device": device
                    }
                    
                    # Use safe file writing with locking
                    if safe_write_to_file(output_file, record):
                        processed_count += 1
                
                # Mark task as done
                work_queue.task_done()
                
            except queue.Empty:
                consecutive_timeouts += 1
                queue_size = work_queue.qsize()
                print(f"Worker on {device}: Timeout #{consecutive_timeouts} after {timeout}s, queue size: {queue_size}")
                
                if consecutive_timeouts >= max_consecutive_timeouts:
                    print(f"Worker on {device}: {max_consecutive_timeouts} consecutive timeouts, shutting down")
                    break
                elif queue_size == 0:
                    print(f"Worker on {device}: Queue empty, shutting down")
                    break
                else:
                    print(f"Worker on {device}: Queue has {queue_size} items, retrying...")
                    continue
            except Exception as e:
                print(f"Worker on {device}: Unexpected error: {str(e)}")
                break
        
        progress_queue.put(('final', device, processed_count))
        print(f"Worker on {device}: Completed processing {processed_count} keys")
        
    except Exception as e:
        print(f"Worker on {device}: Failed to initialize: {str(e)}")
        progress_queue.put(('error', device, str(e)))

def progress_reporter(progress_queue, num_workers):
    """Collect and report progress from all workers"""
    worker_counts = {}
    workers_finished = 0
    
    print("Progress reporter started")
    
    while workers_finished < num_workers:
        try:
            msg_type, device, data = progress_queue.get(timeout=10)
            
            if msg_type == 'progress':
                worker_counts[device] = data
                total_processed = sum(worker_counts.values())
                active_workers = len([d for d, c in worker_counts.items() if c > 0])
                print(f"Total processed: {total_processed} | Active workers: {active_workers} | {worker_counts}")
                
            elif msg_type == 'final':
                worker_counts[device] = data
                workers_finished += 1
                print(f"Worker {device} finished with {data} keys processed ({workers_finished}/{num_workers} workers done)")
                
            elif msg_type == 'error':
                print(f"Worker {device} encountered error: {data}")
                workers_finished += 1
                
        except queue.Empty:
            # Timeout - print current status if we have any data
            if worker_counts:
                total_processed = sum(worker_counts.values())
                active_workers = len(worker_counts)
                print(f"Status: {total_processed} processed | {active_workers} workers reporting | {worker_counts}")
            else:
                print("Progress reporter: No updates received yet")
    
    print("Progress reporter finished")

def get_all_possible_keys(model_name, total_runs=4):
    """Generate all possible keys that should exist for the given model and runs"""
    # Load data to get all possible master_deal_nos
    data = pd.read_csv(CSV_PATH)
    with open(DISTILL_DEALS_PATH, 'r') as f:
        distill_deals = f.read().splitlines()
    distill_deals = [int(d.strip()) for d in distill_deals if d.strip()]
    data = data[data['master_deal_no'].isin(distill_deals)]
    
    all_possible_keys = set()
    for _, obs in data.iterrows():
        master_deal_no = obs['master_deal_no']
        for run in range(total_runs):
            key = f"{master_deal_no}_{model_name}_run_{run}"
            all_possible_keys.add(key)
    
    return all_possible_keys

def ma_inference_parallel(output_folder, model_name, total_runs=4, num_gpus=None, worker_timeout=120):
    """
    Parallel processing with dynamic work queue
    Workers pull work as they complete tasks for optimal load balancing
    """
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {num_gpus} GPUs")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Single output file for all workers
    output_file = os.path.join(output_folder, "lab.jsonl")
    
    # Get all possible keys
    all_possible_keys = get_all_possible_keys(model_name, total_runs)
    print(f"Total possible keys: {len(all_possible_keys)}")
    
    # Get completed keys from the single file
    completed_keys = load_existing_keys(output_file)
    print(f"Already completed keys: {len(completed_keys)}")
    
    # Find remaining keys
    remaining_keys = list(all_possible_keys - completed_keys)
    print(f"Remaining keys to process: {len(remaining_keys)}")
    
    if not remaining_keys:
        print("No remaining work to do!")
        return
    
    # Create shared work queue and progress queue
    work_queue = mp.JoinableQueue()  # Use JoinableQueue for task_done() support
    progress_queue = mp.Queue()
    
    # Populate work queue with remaining keys
    for key in remaining_keys:
        work_queue.put(key)
    
    print(f"Created work queue with {len(remaining_keys)} keys")
    
    # Start progress reporter in separate thread
    progress_thread = threading.Thread(
        target=progress_reporter, 
        args=(progress_queue, num_gpus)
    )
    progress_thread.daemon = True
    progress_thread.start()
    
    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        device = f"cuda:{gpu_id}"
        
        worker = mp.Process(
            target=dynamic_worker,
            args=(work_queue, progress_queue, output_file, model_name, device, worker_timeout)
        )
        workers.append(worker)
        worker.start()
        print(f"Started worker on {device}")
    
    print(f"All {len(workers)} workers started. Processing...")
    
    # Monitor queue and workers periodically
    start_time = time.time()
    last_queue_size = len(remaining_keys)
    
    try:
        while True:
            time.sleep(30)  # Check every 30 seconds
            current_queue_size = work_queue.qsize()
            elapsed = time.time() - start_time
            processed = last_queue_size - current_queue_size
            
            if current_queue_size == 0:
                print(f"Queue empty after {elapsed:.1f}s! Waiting for workers to finish...")
                break
            
            rate = processed / elapsed if elapsed > 0 else 0
            eta = current_queue_size / rate if rate > 0 else float('inf')
            
            print(f"Queue: {current_queue_size}/{last_queue_size} remaining | "
                  f"Rate: {rate:.2f} keys/sec | ETA: {eta/60:.1f} min")
            
            # Check if any workers are still alive
            alive_workers = sum(1 for w in workers if w.is_alive())
            if alive_workers == 0:
                print("All workers have stopped!")
                break
                
        # Wait for all work to be completed
        print("Waiting for work queue to complete...")
        work_queue.join()
        print("All work completed!")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        print("Shutting down workers...")
        
        # Send shutdown signals to workers (one per worker)
        for _ in range(num_gpus):
            try:
                work_queue.put(None, timeout=5)  # Poison pill with timeout
            except:
                pass
        
        # Wait for workers to finish with individual timeouts
        for i, worker in enumerate(workers):
            print(f"Waiting for worker {i} to finish...")
            worker.join(timeout=60)  # Increased timeout for cleanup
            if worker.is_alive():
                print(f"Force terminating worker {worker.pid}")
                worker.terminate()
                worker.join(timeout=10)
                if worker.is_alive():
                    print(f"Force killing worker {worker.pid}")
                    try:
                        import signal
                        os.kill(worker.pid, signal.SIGKILL)
                    except:
                        pass
        
        # Wait for progress thread to finish
        progress_thread.join(timeout=5)
    
    print(f"All workers completed! Results are in {output_file}")

# Model name mapping for plots
MODEL_NAME_MAPPING = {
    "dd_a2": "Linear DD",
    "dk_a_250": "Rank DD",
    "dd_b2": "Linear DD",
    "dk_b_250": "Rank DD"
}

def plot_mention_rates_by_split(results_folder=None, output_path=None):
    """
    Plot mention rates by split (Target vs Non-Target) for different models
    Load all jsonl files from the results folder using glob
    Creates a single bar plot averaging A and B model results
    """
    if results_folder is None:
        results_folder = DEFAULT_RESULTS_FOLDER
    
    if output_path is None:
        output_path = DEFAULT_PLOT_OUTPUT
    
    # Load split files
    with open(SPLIT_A_PATH, 'r') as f:
        split_a_deals = set(int(line.strip()) for line in f if line.strip())
    
    with open(SPLIT_B_PATH, 'r') as f:
        split_b_deals = set(int(line.strip()) for line in f if line.strip())
    
    print(f"Split A deals: {len(split_a_deals)}")
    print(f"Split B deals: {len(split_b_deals)}")

    with open(DISTILL_DEALS_PATH, 'r') as f:
        distill_deals = set(int(line.strip()) for line in f if line.strip())

    delete_deals = []
    for deal in split_a_deals:
        if deal not in distill_deals:
            delete_deals.append(deal)

    for deal in delete_deals:
        split_a_deals.remove(deal)

    delete_deals = []
    for deal in split_b_deals:
        if deal not in distill_deals:
            delete_deals.append(deal)
    for deal in delete_deals:
        split_b_deals.remove(deal)

    print(f"After filtering, Split A deals: {len(split_a_deals)}")
    print(f"After filtering, Split B deals: {len(split_b_deals)}")
    
    # Find all jsonl files in the results folder
    jsonl_pattern = os.path.join(results_folder, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    print(jsonl_files)
    
    print(f"Found {len(jsonl_files)} jsonl files in {results_folder}")
    for file in jsonl_files:
        print(f"  - {file}")
    
    if not jsonl_files:
        print(f"No jsonl files found in {results_folder}")
        return None
    
    # Load results from all jsonl files
    results = []
    for file_path in jsonl_files:
        print(f"Loading results from {file_path}")
        file_results = 0
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    results.append(data)
                    file_results += 1
                except json.JSONDecodeError:
                    continue
        print(f"  - Loaded {file_results} results from {os.path.basename(file_path)}")
    
    print(f"Total loaded results: {len(results)}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Extract model name from key
    def extract_model_from_key(key):
        parts = key.split('_run_')[0]
        model_part = '_'.join(parts.split('_')[1:])  # Everything after first underscore
        return model_part
    
    df['model'] = df['key'].apply(extract_model_from_key)
    
    # Add split information
    def get_split(master_deal_no):
        if master_deal_no in split_a_deals:
            return 'A'
        elif master_deal_no in split_b_deals:
            return 'B'
        else:
            return 'Unknown'
    
    df['split'] = df['master_deal_no'].apply(get_split)
    
    # Filter out unknown splits
    df = df[df['split'] != 'Unknown']
    
    # Determine target vs non-target split for each model
    def get_target_split(model, split):
        if '_a' in model:  # A models
            return 'Target Split' if split == 'A' else 'Non-Target Split'
        else:  # B models
            return 'Target Split' if split == 'B' else 'Non-Target Split'
    
    df['target_split'] = df.apply(lambda row: get_target_split(row['model'], row['split']), axis=1)
    
    # Calculate mention rates by model and target split
    rates = df
    rates['model_display'] = rates['model'].map(MODEL_NAME_MAPPING)
    rates = rates[rates['model_display'].notnull()]  # Filter out unmapped models
    rates['demeaned_mention_detected'] = rates.groupby(['model_display', 'target_split'])['mention_detected'].transform(lambda x: x - x.mean())
    
    population_standard_error = rates['demeaned_mention_detected'].std() / (len(rates) ** 0.5)

    # Create single bar plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Set custom order to put Linear DD first
    model_order = ['Linear DD', 'Rank DD']
    
    sns.barplot(
        data=rates,
        x='model_display',
        y='mention_detected',
        hue='target_split',
        saturation=1,
        order=model_order,
        ax=ax,
        errorbar=(lambda x: (x.mean() - 2.576*population_standard_error, min(x.mean() + 2.576*population_standard_error, 1)))
    )
    
    ax.set_xlabel('')
    ax.set_ylabel('Target Firm Mention Rate (%)')
    ax.set_ylim(0, 1)
    ax.set_yticklabels([f"{int(tick*100)}%" for tick in ax.get_yticks()])
    ax.legend(title='', loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_path}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    model_names = ["dd_a2", "dd_b2", "dk_a_250", "dk_b_250"]
    # for model_name in model_names:
    #     print(f"Processing model: {model_name}")
    #     ma_inference_parallel(
    #         output_folder=DEFAULT_OUTPUT_FOLDER,
    #         model_name=model_name,
    #         total_runs=1,
    #         num_gpus=None,  # Auto-detect GPUs
    #         worker_timeout=300 
    #     )

    print("Creating mention rates plot...")
    plot_mention_rates_by_split("")