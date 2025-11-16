#https://github.com/EleutherAI/lm-evaluation-harness
import os
from dotenv import load_dotenv
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

#("--model two-model-gemma --model_args topk=250,run_index=50", "topk250_ma")
Run_Models = [("--model two-model-gemma --model_args alpha=2,run_index=50", "alpha2_ma")]

Run_Tests = [("mmlu --num_fewshot 0", "mmlu_cot_0shot", "mmlu")]

scores = []

for model_command, model_name in Run_Models:
    for task_command, task_name, json_dir in Run_Tests:
        directory = f"model_eval/{model_name}/{task_name}/"
        if os.path.isdir(directory):
            print(f"Already done with {task_name}, model {model_name}")
        else:
            start_time = time.time()
            command = f"lm_eval {model_command} --tasks {task_command} --output_path model_eval/{model_name}/{task_name}/ --batch_size auto"
            print(command)
            os.system(command)
            end_time = time.time()
            print(f"Done with task {task_name}, model {model_name}, took {end_time - start_time:.2f} seconds.")

        #get score
        try:
            json_file = os.listdir(directory)[0]
            with open(f"{directory}{json_file}") as f:
                data = json.load(f)
            score = data["results"][json_dir]["exact_match,get_response"]
            name = data["results"][json_dir]["alias"]
            scores.append({"Model Name": model_name, "Task Name": name, "Score": score})
        except Exception as e:
            print(f"Exception, passing")