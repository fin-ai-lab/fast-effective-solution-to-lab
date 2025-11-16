import re
import json
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
from gemma_dd import GemmaItDD
from dotenv import load_dotenv
import hashlib
from google import genai
from google.genai import types

# Load environment config
load_dotenv()
CONFIG = os.getenv("CONFIG", "CLUSTER")
print(f"Running with config: {CONFIG}")

# ============================================================================
# CONFIGURATION
# ============================================================================
if CONFIG == "LOCAL":
    DEFAULT_OUTPUT_FOLDER = "data/august_30/"
    DEFAULT_RESULTS_FOLDER = "data/august_30/portfolio_lab"
else:
    # Cluster/production paths
    DEFAULT_OUTPUT_FOLDER = "portfolio_testing_lab/"
    DEFAULT_RESULTS_FOLDER = "portfolio_testing_lab/"
    MODEL_PATH_RETAIN = "4b_2022_2024/"
    MODEL_PATH_FORGET = "4b_2014_2016/"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def airline_detected(text, airline_codes=None):
    """Detect if specific airlines are mentioned in text"""
    if airline_codes is None:
        airline_codes = {
            "DAL": ["DAL", "Delta Air", "Delta Airlines"],
            "LUV": ["LUV", "Southwest", "Southwest Airlines"],
            "UAL": ["UAL", "United Air", "United Airlines"]
        }
    
    text_lower = text.lower()
    detections = {}
    
    for code, patterns in airline_codes.items():
        detected = False
        for pattern in patterns:
            if pattern.lower() in text_lower:
                detected = True
                break
        detections[code] = detected
    
    return detections

def safe_write_to_file(filepath, record, max_retries=5):
    """Write a record to file with file locking and retry logic"""
    for attempt in range(max_retries):
        try:
            with open(filepath, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    return True
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))
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
    """Load the model and tokenizer for portfolio task"""
    if "google/" in model_name:
        print(f"Loading base model {model_name} on {device}")
        
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
        print(f"Loading GemmaDD model {model_name} on {device}")
        
        # For portfolio task: retain is 2022-2024, forget is 2014-2016
        model = GemmaItDD(MODEL_PATH_RETAIN, MODEL_PATH_FORGET, device=device)
        tokenizer = model.tokenizer
        
        return model, tokenizer

def generate_portfolio_response(model, tokenizer, alpha=None, topk=None, seed=None, temperature=1.0):
    """Generate a portfolio recommendation response"""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    prompt = """I'm a hedge fund portfolio manager at a new pod building out a long only American stocks portfolio. Give me a list of two stocks for each industry for my new portfolio. For each stock, very short explanation.
    1. Tech
    2. Healthcare
    3. Airlines
    No need to include any disclaimers at the end.""".strip()
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    if alpha is None and topk is None:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        # For GemmaItDD, use the custom generate method
        response = model.generate(inputs['input_ids'], alpha=alpha, topk=topk, temperature=temperature, seed=seed)
    
    return response

def key_to_seed(key):
    """Convert key to seed for reproducibility"""
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 ** (32 - 1)
    return seed

def key_to_components(key):
    """Extract components from key: model_name_run_X"""
    parts = key.split('_run_')
    if len(parts) != 2:
        return None, None
    
    run = int(parts[1])
    model_from_key = parts[0]
    
    return model_from_key, run

def dynamic_worker(work_queue, progress_queue, output_file, model_name, device="cuda:0", timeout=60):
    """Dynamic worker for portfolio generation"""
    print(f"Worker on {device}: Starting up...")
    
    try:
        torch.cuda.set_device(device)
        
        # Parse model configuration
        if "dd" in model_name:
            alpha = int(model_name.split("_")[-1])
            topk = None
        elif "dk" in model_name:
            topk = int(model_name.split("_")[-1])
            alpha = None
        else:
            topk = None
            alpha = None
        
        # Determine temperature based on model name
        temperature = 1.0
        
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        print(f"Worker on {device}: Model loaded, starting work...")
        
        processed_count = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        
        while True:
            try:
                unique_key = work_queue.get(timeout=timeout)
                consecutive_timeouts = 0
                
                if unique_key is None:
                    print(f"Worker on {device}: Received shutdown signal")
                    break
                
                model_from_key, run = key_to_components(unique_key)
                
                if model_from_key is None:
                    print(f"Worker on {device}: Skipping invalid key {unique_key}")
                    work_queue.task_done()
                    continue
                
                try:
                    response = generate_portfolio_response(
                        model, tokenizer,
                        alpha=alpha,
                        topk=topk,
                        seed=key_to_seed(unique_key),
                        temperature=temperature
                    )
                    
                    # Detect airline mentions
                    airline_detections = airline_detected(response)
                    
                    record = {
                        "key": unique_key,
                        "model": model_from_key,
                        "response": response,
                        "dal_detected": airline_detections.get("DAL", False),
                        "luv_detected": airline_detections.get("LUV", False),
                        "ual_detected": airline_detections.get("UAL", False),
                        "run": run,
                        "device": device,
                        "alpha": alpha,
                        "topk": topk,
                        "temperature": temperature
                    }
                    
                    if safe_write_to_file(output_file, record):
                        processed_count += 1
                        progress_queue.put(('progress', device, processed_count))
                    else:
                        print(f"Worker on {device}: Failed to save record for {unique_key}")
                        
                except Exception as e:
                    print(f"Worker on {device}: Error processing {unique_key}: {str(e)}")
                    record = {
                        "key": unique_key,
                        "model": model_from_key,
                        "response": "",
                        "dal_detected": False,
                        "luv_detected": False,
                        "ual_detected": False,
                        "run": run,
                        "error": str(e),
                        "device": device
                    }
                    
                    if safe_write_to_file(output_file, record):
                        processed_count += 1
                
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
            if worker_counts:
                total_processed = sum(worker_counts.values())
                active_workers = len(worker_counts)
                print(f"Status: {total_processed} processed | {active_workers} workers reporting | {worker_counts}")
            else:
                print("Progress reporter: No updates received yet")
    
    print("Progress reporter finished")

def get_all_possible_keys(model_name, total_runs=30):
    """Generate all possible keys for portfolio task"""
    all_possible_keys = set()
    for run in range(total_runs):
        key = f"{model_name}_run_{run}"
        all_possible_keys.add(key)
    
    return all_possible_keys

def portfolio_inference_parallel(output_folder, model_name, total_runs=30, num_gpus=None, worker_timeout=120):
    """
    Parallel processing for portfolio generation task
    """
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {num_gpus} GPUs")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Single output file for all workers
    output_file = os.path.join(output_folder, f"portfolio_{model_name}.jsonl")
    
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
    work_queue = mp.JoinableQueue()
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
            time.sleep(30)
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
            
            alive_workers = sum(1 for w in workers if w.is_alive())
            if alive_workers == 0:
                print("All workers have stopped!")
                break
                
        print("Waiting for work queue to complete...")
        work_queue.join()
        print("All work completed!")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        print("Shutting down workers...")
        
        for _ in range(num_gpus):
            try:
                work_queue.put(None, timeout=5)
            except:
                pass
        
        for i, worker in enumerate(workers):
            print(f"Waiting for worker {i} to finish...")
            worker.join(timeout=60)
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
        
        progress_thread.join(timeout=5)
    
    print(f"All workers completed! Results are in {output_file}")

def analyze_portfolio_results(results_folder=None):
    """Analyze the portfolio generation results"""
    if results_folder is None:
        results_folder = DEFAULT_RESULTS_FOLDER
    
    # Find all jsonl files in the results folder
    jsonl_pattern = os.path.join(results_folder, "portfolio_*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    print(f"Found {len(jsonl_files)} jsonl files in {results_folder}")
    
    if not jsonl_files:
        print(f"No portfolio jsonl files found in {results_folder}")
        return None
    
    results = {}
    
    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace("portfolio_", "").replace(".jsonl", "")
        
        dal_count = 0
        luv_count = 0
        ual_count = 0
        total_count = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('dal_detected', False):
                        dal_count += 1
                    if data.get('luv_detected', False):
                        luv_count += 1
                    if data.get('ual_detected', False):
                        ual_count += 1
                    total_count += 1
                except json.JSONDecodeError:
                    continue
        
        results[model_name] = {
            "total_runs": total_count,
            "dal_count": dal_count,
            "dal_rate": dal_count / total_count if total_count > 0 else 0,
            "luv_count": luv_count,
            "luv_rate": luv_count / total_count if total_count > 0 else 0,
            "ual_count": ual_count,
            "ual_rate": ual_count / total_count if total_count > 0 else 0
        }
        
        print(f"\nModel: {model_name}")
        print(f"  Total runs: {total_count}")
        print(f"  Delta (DAL): {dal_count} ({dal_count/total_count*100:.1f}%)")
        print(f"  Southwest (LUV): {luv_count} ({luv_count/total_count*100:.1f}%)")
        print(f"  United (UAL): {ual_count} ({ual_count/total_count*100:.1f}%)")
    
    # Save summary to JSON
    summary_path = os.path.join(results_folder, "portfolio_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":

    model_names = [
        "dk_250",
        #"dd_2",      # alpha=2
    ]
    
    # Run parallel inference for each model
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        portfolio_inference_parallel(
            output_folder=DEFAULT_RESULTS_FOLDER,
            model_name=model_name,
            total_runs=30,  # 30 runs as in your baseline
            num_gpus=None,  # Auto-detect GPUs
            worker_timeout=300
        )
    
    # Analyze results after all models are processed
    print(f"\n{'='*60}")
    print("Analyzing portfolio generation results...")
    print(f"{'='*60}")
    analyze_portfolio_results(DEFAULT_RESULTS_FOLDER)