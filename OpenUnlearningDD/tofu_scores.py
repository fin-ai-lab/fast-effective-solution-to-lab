import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

privacy_keys = ["mia_min_k_plus_plus", "mia_min_k", "mia_loss", "mia_zlib"]
memorization_keys = ["extraction_strength", "exact_memorization", "forget_Q_A_PARA_Prob", "forget_truth_ratio"]
utility_keys = ["model_utility", "forget_Q_A_gibberish"]

retrain_priv_scores = {}
with open("saves/eval/tofu_retrain/TOFU_SUMMARY.json", "r") as f:
    data = json.load(f)
    for key in privacy_keys:
        retrain_priv_scores[key] = data[key]

with open("saves/eval/tofu_target/TOFU_SUMMARY.json", "r") as f:
    data = json.load(f)
    target_util = 2 / ((1 / data["model_utility"]) + (1 / data["forget_Q_A_gibberish"]))

def view_pretty(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    print(df.to_string(index=False))

def calculate_info(directory):
    with open(directory, "r") as f:
        data = json.load(f)

    memorization_score = len(memorization_keys) / sum(1 / (1-data[key]) for key in memorization_keys)
    try:
        utility_score = (len(utility_keys) / sum(1 / data[key] for key in utility_keys)) / target_util
    except ZeroDivisionError:
        utility_score = 0.0
    agg_wo_privacy = 2 / ((1 / (memorization_score)) + (1 / (utility_score))) if utility_score > 0 else 0.0

    if "mia_min_k_plus_plus" in data:
        privacy_score = len(privacy_keys) / sum(1 / (1 - math.fabs(data[key] - retrain_priv_scores[key])) for key in privacy_keys)
        agg = 3 / ((1 / (memorization_score)) + (1 / (privacy_score)) + (1 / (utility_score))) if utility_score > 0 else 0.0
    else:
        agg = float('nan')
        privacy_score = float('nan')

    results = {
        "agg": agg,
        "agg_wo_privacy": agg_wo_privacy,
        "memorization_score": memorization_score,
        "privacy_score": privacy_score,
        "utility_score": utility_score
    }

    return results

def find_optimal_dd_configs():
    """Find optimal DD configurations for both linear and rank methods"""
    optimal_configs = {}
    
    # Linear DD configurations
    for model_size in ["3.2-1B", "3.2-3B"]:
        best_agg = -1
        best_agg_wo_priv = -1
        best_alpha_agg = None
        best_alpha_agg_wo_priv = None
        
        for alpha in [0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0]:
            try:
                directory = f"saves/eval/tofu_linear/alpha-{alpha}-{model_size}/TOFU_SUMMARY.json"
                info = calculate_info(directory)
                
                # Check for best aggregate score (with privacy)
                if not math.isnan(info["agg"]) and info["agg"] > best_agg:
                    best_agg = info["agg"]
                    best_alpha_agg = alpha
                
                # Check for best aggregate score without privacy
                if info["agg_wo_privacy"] > best_agg_wo_priv:
                    best_agg_wo_priv = info["agg_wo_privacy"]
                    best_alpha_agg_wo_priv = alpha
                    
            except FileNotFoundError:
                pass
        
        optimal_configs[f"Linear DD {model_size}"] = {
            "best_alpha_agg": best_alpha_agg,
            "best_alpha_agg_wo_priv": best_alpha_agg_wo_priv
        }
    
    # Rank DD configurations
    for model_size in ["3.2-1B", "3.2-3B"]:
        best_agg = -1
        best_agg_wo_priv = -1
        best_topk_agg = None
        best_topk_agg_wo_priv = None
        
        for topk in [1, 3, 5, 10, 20, 100]:
            try:
                directory = f"saves/eval/tofu_rank/topk-{topk}-{model_size}/TOFU_SUMMARY.json"
                info = calculate_info(directory)
                
                # Check for best aggregate score (with privacy)
                if not math.isnan(info["agg"]) and info["agg"] > best_agg:
                    best_agg = info["agg"]
                    best_topk_agg = topk
                
                # Check for best aggregate score without privacy
                if info["agg_wo_privacy"] > best_agg_wo_priv:
                    best_agg_wo_priv = info["agg_wo_privacy"]
                    best_topk_agg_wo_priv = topk
                    
            except FileNotFoundError:
                pass
        
        optimal_configs[f"Rank DD {model_size}"] = {
            "best_topk_agg": best_topk_agg,
            "best_topk_agg_wo_priv": best_topk_agg_wo_priv
        }
    
    return optimal_configs

def find_optimal_unlearning_configs():
    """Find optimal configurations for unlearning methods with learning rate search"""
    unlearn_methods = ["DPO", "GradAscent", "GradDiff", "NPO", "RMU"]
    checkpoints = [13, 26, 39, 52, 65, 78, 91, 104, 117, 130]
    learning_rates = ["2e-6", "1e-6", "3e-6", "1.5e-6", "4e-6", "8e-7"]  # Learning rate strings as they appear in filenames
    optimal_configs = {}
    
    for method in unlearn_methods:
        best_agg = -1
        best_agg_wo_priv = -1
        best_config_agg = None
        best_config_agg_wo_priv = None
        
        # Search over learning rates and checkpoints
        for lr in learning_rates:
            for checkpoint in checkpoints:
                try:
                    method_dir = f"unlearn/tofu/tofu_Llama-3.1-8B-Instruct_forget10_{method}_{lr}/checkpoint-{checkpoint}/evals"
                    info = calculate_info(f"saves/{method_dir}/TOFU_SUMMARY.json")
                    
                    print(f"Method: {method}, LR: {lr}, Checkpoint: {checkpoint}, Agg: {info['agg']}, Agg w/o Priv: {info['agg_wo_privacy']}")

                    # Check for best aggregate score (with privacy)
                    if not math.isnan(info["agg"]) and info["agg"] > best_agg:
                        best_agg = info["agg"]
                        best_config_agg = {
                            "checkpoint": checkpoint,
                            "learning_rate": lr,
                            "epoch": int(checkpoint / 13)
                        }
                    
                    # Check for best aggregate score without privacy
                    if info["agg_wo_privacy"] > best_agg_wo_priv:
                        best_agg_wo_priv = info["agg_wo_privacy"]
                        best_config_agg_wo_priv = {
                            "checkpoint": checkpoint,
                            "learning_rate": lr,
                            "epoch": int(checkpoint / 13)
                        }
                        
                except (FileNotFoundError, KeyError):
                    pass
        
        optimal_configs[method] = {
            "best_config_agg": best_config_agg,
            "best_config_agg_wo_priv": best_config_agg_wo_priv
        }
    
    return optimal_configs

def find_top_two_values(df, column, exclude_indices=[0, 1]):
    """Find indices of top two values in a column, excluding specified indices"""
    valid_data = []
    for i, val in enumerate(df[column]):
        if i not in exclude_indices and not pd.isna(val):
            valid_data.append((i, val))
    
    # Sort by value in descending order
    valid_data.sort(key=lambda x: x[1], reverse=True)
    
    # Return indices of top two
    top_indices = [x[0] for x in valid_data[:2]]
    return top_indices

def generate_latex_table(df, table_name, with_privacy=True):
    """Generate LaTeX table format with bold highlighting for top two values"""
    print(f"\n{table_name} - LaTeX Format:")
    print("="*60)
    
    # Start table
    if with_privacy:
        print("\\begin{tabular}{lcccccc}")
        print("\\toprule")
        print("Method & Config & Agg. $\\uparrow$ & Mem. $\\uparrow$ & Priv. $\\uparrow$ & Utility $\\uparrow$ \\\\")
        columns_to_highlight = ['Agg', 'Mem', 'Priv', 'Utility']
    else:
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Method & Config & Agg. w/o Priv. $\\uparrow$ & Mem. $\\uparrow$ & Utility $\\uparrow$ \\\\")
        columns_to_highlight = ['Agg w/o Priv', 'Mem', 'Utility']
    
    print("\\midrule")
    
    # Find top two indices for each column (excluding target and retrain - indices 0 and 1)
    top_indices = {}
    for col in columns_to_highlight:
        top_indices[col] = find_top_two_values(df, col, exclude_indices=[0, 1])
    
    # Add rows
    for i, row in df.iterrows():
        if with_privacy:
            # Handle NaN values and apply bold formatting
            priv_val = "N/A" if pd.isna(row['Priv']) else f"{row['Priv']:.2f}"
            agg_val = "N/A" if pd.isna(row['Agg']) else f"{row['Agg']:.2f}"
            
            # Apply bold formatting for top values
            if i in top_indices['Agg'] and not pd.isna(row['Agg']):
                agg_val = f"\\textbf{{{agg_val}}}"
            if i in top_indices['Mem']:
                mem_val = f"\\textbf{{{row['Mem']:.2f}}}"
            else:
                mem_val = f"{row['Mem']:.2f}"
            if i in top_indices['Priv'] and not pd.isna(row['Priv']):
                priv_val = f"\\textbf{{{priv_val}}}"
            if i in top_indices['Utility']:
                util_val = f"\\textbf{{{row['Utility']:.2f}}}"
            else:
                util_val = f"{row['Utility']:.2f}"
            
            print(f"{row['Method']} & {row['Config']} & {agg_val} & {mem_val} & {priv_val} & {util_val} \\\\")
        else:
            # Apply bold formatting for top values
            if i in top_indices['Agg w/o Priv']:
                agg_val = f"\\textbf{{{row['Agg w/o Priv']:.2f}}}"
            else:
                agg_val = f"{row['Agg w/o Priv']:.2f}"
            if i in top_indices['Mem']:
                mem_val = f"\\textbf{{{row['Mem']:.2f}}}"
            else:
                mem_val = f"{row['Mem']:.2f}"
            if i in top_indices['Utility']:
                util_val = f"\\textbf{{{row['Utility']:.2f}}}"
            else:
                util_val = f"{row['Utility']:.2f}"
            
            print(f"{row['Method']} & {row['Config']} & {agg_val} & {mem_val} & {util_val} \\\\")
        
        # Add midrule after baselines
        if i == 1:  # After retrain row
            print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")

def generate_tofu_tables():
    """Generate the TOFU tables with optimal configurations"""
    print("Finding optimal configurations...")
    dd_configs = find_optimal_dd_configs()
    unlearn_configs = find_optimal_unlearning_configs()
    
    # Table 1: With Privacy (includes Linear DD, Rank DD, and unlearning methods)
    table1_data = []
    
    # Add Target and Retrain baselines
    target_info = calculate_info("saves/eval/tofu_target/TOFU_SUMMARY.json")
    retrain_info = calculate_info("saves/eval/tofu_retrain/TOFU_SUMMARY.json")
    
    table1_data.append({
        "Method": "Target",
        "Config": "Full",
        "Agg": target_info["agg"],
        "Mem": target_info["memorization_score"],
        "Priv": target_info["privacy_score"],
        "Utility": target_info["utility_score"]
    })
    
    table1_data.append({
        "Method": "Retrain",
        "Config": "Retain90",
        "Agg": retrain_info["agg"],
        "Mem": retrain_info["memorization_score"],
        "Priv": retrain_info["privacy_score"],
        "Utility": retrain_info["utility_score"]
    })
    
    # Add Linear DD methods
    for model_size in ["3.2-1B", "3.2-3B"]:
        config_key = f"Linear DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_alpha_agg"] is not None:
            alpha = dd_configs[config_key]["best_alpha_agg"]
            directory = f"saves/eval/tofu_linear/alpha-{alpha}-{model_size}/TOFU_SUMMARY.json"
            try:
                info = calculate_info(directory)
                table1_data.append({
                    "Method": f"Linear DD {model_size.split('-')[1]}",
                    "Config": f"$\\alpha$={alpha}",
                    "Agg": info["agg"],
                    "Mem": info["memorization_score"],
                    "Priv": info["privacy_score"],
                    "Utility": info["utility_score"]
                })
            except FileNotFoundError:
                pass
    
    # Add Rank DD methods
    for model_size in ["3.2-1B", "3.2-3B"]:
        config_key = f"Rank DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_topk_agg"] is not None:
            topk = dd_configs[config_key]["best_topk_agg"]
            directory = f"saves/eval/tofu_rank/topk-{topk}-{model_size}/TOFU_SUMMARY.json"
            try:
                info = calculate_info(directory)
                table1_data.append({
                    "Method": f"Rank DD {model_size.split('-')[1]}",
                    "Config": f"topk={topk}",
                    "Agg": info["agg"],
                    "Mem": info["memorization_score"],
                    "Priv": info["privacy_score"],
                    "Utility": info["utility_score"]
                })
            except FileNotFoundError:
                pass
    
    # Add unlearning methods
    for method in ["DPO", "GradAscent", "GradDiff", "NPO", "RMU"]:
        if method in unlearn_configs and unlearn_configs[method]["best_config_agg"] is not None:
            config = unlearn_configs[method]["best_config_agg"]
            checkpoint = config["checkpoint"]
            lr = config["learning_rate"]
            epoch = config["epoch"]
            
            method_dir = f"unlearn/tofu/tofu_Llama-3.1-8B-Instruct_forget10_{method}_{lr}/checkpoint-{checkpoint}/evals"
            try:
                info = calculate_info(f"saves/{method_dir}/TOFU_SUMMARY.json")
                table1_data.append({
                    "Method": method,
                    "Config": f"lr={lr}, epoch={epoch}",
                    "Agg": info["agg"],
                    "Mem": info["memorization_score"],
                    "Priv": info["privacy_score"],
                    "Utility": info["utility_score"]
                })
            except (FileNotFoundError, KeyError):
                pass
    
    df1 = pd.DataFrame(table1_data)
    
    # Table 2: Without Privacy (includes Linear DD, Rank DD, and unlearning methods) - unchanged
    table2_data = []
    
    # Add Target and Retrain baselines
    table2_data.append({
        "Method": "Target",
        "Config": "Full",
        "Agg w/o Priv": target_info["agg_wo_privacy"],
        "Mem": target_info["memorization_score"],
        "Utility": target_info["utility_score"]
    })
    
    table2_data.append({
        "Method": "Retrain",
        "Config": "Retain90",
        "Agg w/o Priv": retrain_info["agg_wo_privacy"],
        "Mem": retrain_info["memorization_score"],
        "Utility": retrain_info["utility_score"]
    })
    
    # Add Linear DD methods
    for model_size in ["3.2-1B", "3.2-3B"]:
        config_key = f"Linear DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_alpha_agg_wo_priv"] is not None:
            alpha = dd_configs[config_key]["best_alpha_agg_wo_priv"]
            directory = f"saves/eval/tofu_linear/alpha-{alpha}-{model_size}/TOFU_SUMMARY.json"
            try:
                info = calculate_info(directory)
                table2_data.append({
                    "Method": f"Linear DD {model_size.split('-')[1]}",
                    "Config": f"$\\alpha$={alpha}",
                    "Agg w/o Priv": info["agg_wo_privacy"],
                    "Mem": info["memorization_score"],
                    "Utility": info["utility_score"]
                })
            except FileNotFoundError:
                pass
    
    # Add Rank DD methods
    for model_size in ["3.2-1B", "3.2-3B"]:
        config_key = f"Rank DD {model_size}"
        if config_key in dd_configs and dd_configs[config_key]["best_topk_agg_wo_priv"] is not None:
            topk = dd_configs[config_key]["best_topk_agg_wo_priv"]
            directory = f"saves/eval/tofu_rank/topk-{topk}-{model_size}/TOFU_SUMMARY.json"
            try:
                info = calculate_info(directory)
                table2_data.append({
                    "Method": f"Rank DD {model_size.split('-')[1]}",
                    "Config": f"topk={topk}",
                    "Agg w/o Priv": info["agg_wo_privacy"],
                    "Mem": info["memorization_score"],
                    "Utility": info["utility_score"]
                })
            except FileNotFoundError:
                pass
    
    # Add unlearning methods
    for method in ["DPO", "GradAscent", "GradDiff", "NPO", "RMU"]:
        if method in unlearn_configs and unlearn_configs[method]["best_config_agg_wo_priv"] is not None:
            config = unlearn_configs[method]["best_config_agg_wo_priv"]
            checkpoint = config["checkpoint"]
            lr = config["learning_rate"]
            epoch = config["epoch"]
            
            method_dir = f"unlearn/tofu/tofu_Llama-3.1-8B-Instruct_forget10_{method}_{lr}/checkpoint-{checkpoint}/evals"
            try:
                info = calculate_info(f"saves/{method_dir}/TOFU_SUMMARY.json")
                table2_data.append({
                    "Method": method,
                    "Config": f"lr={lr}, epoch={epoch}",
                    "Agg w/o Priv": info["agg_wo_privacy"],
                    "Mem": info["memorization_score"],
                    "Utility": info["utility_score"]
                })
            except (FileNotFoundError, KeyError):
                pass
    
    df2 = pd.DataFrame(table2_data)
    
    # Generate LaTeX tables
    generate_latex_table(df1, "TABLE 1: TOFU Results with Privacy - Agg. is harmonic mean", with_privacy=True)
    generate_latex_table(df2, "TABLE 2: TOFU Results without Privacy", with_privacy=False)
    
    return df1, df2

# Generate the tables
generate_tofu_tables()