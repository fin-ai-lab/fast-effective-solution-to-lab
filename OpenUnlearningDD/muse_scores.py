import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# SNS tab10 color palette - using same colors for same model sizes
palette = {
    "Target": "#1f77b4",
    "Retrain": "#ff7f0e",
    #"GradAscent": "#e377c2",
    "GradDiff": "#bcbd22",
    "NPO": "#9467bd",
    "SimNPO": "#2ca02c",
    "Linear DD": "#17becf", 
    "Rank DD": "#d62728",      
    # "Trigram Rank DD": "#7f7f7f", 
    # "Trigram Linear DD": "#8c564b",
}

# Different markers for linear vs rank methods
markers = {
    "Target": "s",
    "Retrain": "s",
    "GradDiff": "o",
    "NPO": "o",
    "SimNPO": "o",
    "Linear DD": "X",         
    "Rank DD": "X",            
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

alpha_values_model = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
topk_values_model = [1, 5, 20, 50, 100, 200, 500, 1000]
alpha_values_trigram = [5, 10, 15, 20, 25, 30]
topk_values_trigram = [1, 2, 3, 5, 10]

def find_optimal_configurations():
    """
    Find the optimal alpha and topk configurations for each model size.
    Now finds separate optimal configurations for verbmem and knowmem metrics.
    Returns a dictionary with the optimal settings for hardcoding.
    """
    
    # Load baselines for distance calculation
    retrain_target = ["Target", "Retrain"]
    baseline_scores = {}
    
    for name in retrain_target:
        info = json.load(open(f"saves/eval/muse_main/muse_{name.lower()}/MUSE_SUMMARY.json"))
        baseline_scores[name] = {
            'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
            'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
            'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
        }
    
    retrain_scores = baseline_scores['Retrain']
    
    def calculate_distance_verbmem(point):
        """Calculate euclidean distance from retrain baseline using verbmem and retain metrics"""
        forget_verbmem_diff = point['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        
        return (forget_verbmem_diff**2 + retain_diff**2)**0.5
    
    def calculate_distance_knowmem(point):
        """Calculate euclidean distance from retrain baseline using knowmem and retain metrics"""
        forget_knowmem_diff = point['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        
        return (forget_knowmem_diff**2 + retain_diff**2)**0.5
    
    model_sizes = ["1.3b", "2.7b", "Trigram", ] #"7b"
    optimal_configs = {}
    
    print("FINDING OPTIMAL CONFIGURATIONS (SEPARATE FOR VERBMEM AND KNOWMEM)")
    print("="*80)
    
    for model_size in model_sizes:
        print(f"\nProcessing {model_size}...")
        optimal_configs[model_size] = {}
        
        # Process alpha-based models
        alpha_data = []
        if model_size == "Trigram":
            alpha_values = alpha_values_trigram
            topk_values = topk_values_trigram
        else:
            alpha_values = alpha_values_model
            topk_values = topk_values_model

        for alpha in alpha_values:
            folder_name = f"muse_main/muse-{model_size}-alpha-{alpha}"
            info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
            alpha_data.append({
                'alpha': alpha,
                'folder': folder_name,
                'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            })
        
        # Process topk-based models
        topk_data = []
        for topk in topk_values:
            folder_name = f"muse_main/muse-{model_size}-topk-{topk}"
            info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
            topk_data.append({
                'topk': topk,
                'folder': folder_name,
                'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            })
        
        # Find optimal alpha for verbmem
        if alpha_data:
            best_alpha_distance_verbmem = float('inf')
            best_alpha_config_verbmem = None
            
            for point in alpha_data:
                distance = calculate_distance_verbmem(point)
                if distance < best_alpha_distance_verbmem:
                    best_alpha_distance_verbmem = distance
                    best_alpha_config_verbmem = point
            
            if best_alpha_config_verbmem:
                optimal_configs[model_size]['alpha_verbmem'] = {
                    'value': best_alpha_config_verbmem['alpha'],
                    'folder': best_alpha_config_verbmem['folder'],
                    'distance': best_alpha_distance_verbmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_alpha_config_verbmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_alpha_config_verbmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_alpha_config_verbmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best alpha for verbmem: {best_alpha_config_verbmem['alpha']} (distance: {best_alpha_distance_verbmem:.4f})")
        
        # Find optimal alpha for knowmem
        if alpha_data:
            best_alpha_distance_knowmem = float('inf')
            best_alpha_config_knowmem = None
            
            for point in alpha_data:
                distance = calculate_distance_knowmem(point)
                if distance < best_alpha_distance_knowmem:
                    best_alpha_distance_knowmem = distance
                    best_alpha_config_knowmem = point
            
            if best_alpha_config_knowmem:
                optimal_configs[model_size]['alpha_knowmem'] = {
                    'value': best_alpha_config_knowmem['alpha'],
                    'folder': best_alpha_config_knowmem['folder'],
                    'distance': best_alpha_distance_knowmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_alpha_config_knowmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_alpha_config_knowmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_alpha_config_knowmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best alpha for knowmem: {best_alpha_config_knowmem['alpha']} (distance: {best_alpha_distance_knowmem:.4f})")
        
        # Find optimal topk for verbmem
        if topk_data:
            best_topk_distance_verbmem = float('inf')
            best_topk_config_verbmem = None
            
            for point in topk_data:
                distance = calculate_distance_verbmem(point)
                if distance < best_topk_distance_verbmem:
                    best_topk_distance_verbmem = distance
                    best_topk_config_verbmem = point
            
            if best_topk_config_verbmem:
                optimal_configs[model_size]['topk_verbmem'] = {
                    'value': best_topk_config_verbmem['topk'],
                    'folder': best_topk_config_verbmem['folder'],
                    'distance': best_topk_distance_verbmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_topk_config_verbmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_topk_config_verbmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_topk_config_verbmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best topk for verbmem: {best_topk_config_verbmem['topk']} (distance: {best_topk_distance_verbmem:.4f})")
        
        # Find optimal topk for knowmem
        if topk_data:
            best_topk_distance_knowmem = float('inf')
            best_topk_config_knowmem = None
            
            for point in topk_data:
                distance = calculate_distance_knowmem(point)
                if distance < best_topk_distance_knowmem:
                    best_topk_distance_knowmem = distance
                    best_topk_config_knowmem = point
            
            if best_topk_config_knowmem:
                optimal_configs[model_size]['topk_knowmem'] = {
                    'value': best_topk_config_knowmem['topk'],
                    'folder': best_topk_config_knowmem['folder'],
                    'distance': best_topk_distance_knowmem,
                    'scores': {
                        'forget_verbmem_ROUGE': best_topk_config_knowmem['forget_verbmem_ROUGE'],
                        'forget_knowmem_ROUGE': best_topk_config_knowmem['forget_knowmem_ROUGE'],
                        'retain_knowmem_ROUGE': best_topk_config_knowmem['retain_knowmem_ROUGE']
                    }
                }
                print(f"  Best topk for knowmem: {best_topk_config_knowmem['topk']} (distance: {best_topk_distance_knowmem:.4f})")
    
    # Print detailed summary for hardcoding
    print(f"\n{'='*80}")
    print("OPTIMAL CONFIGURATIONS FOR HARDCODING")
    print(f"{'='*80}")
    
    for model_size, configs in optimal_configs.items():
        print(f"\n{model_size.upper()}:")
        if 'alpha_verbmem' in configs:
            alpha_config = configs['alpha_verbmem']
            print(f"  Linear DD (alpha) for verbmem: {alpha_config['value']} -> folder: {alpha_config['folder']}")
            scores = alpha_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
        
        if 'alpha_knowmem' in configs:
            alpha_config = configs['alpha_knowmem']
            print(f"  Linear DD (alpha) for knowmem: {alpha_config['value']} -> folder: {alpha_config['folder']}")
            scores = alpha_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
        
        if 'topk_verbmem' in configs:
            topk_config = configs['topk_verbmem']
            print(f"  Rank DD (topk) for verbmem: {topk_config['value']} -> folder: {topk_config['folder']}")
            scores = topk_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
        
        if 'topk_knowmem' in configs:
            topk_config = configs['topk_knowmem']
            print(f"  Rank DD (topk) for knowmem: {topk_config['value']} -> folder: {topk_config['folder']}")
            scores = topk_config['scores']
            print(f"    Scores: verbmem={scores['forget_verbmem_ROUGE']:.2f}%, knowmem={scores['forget_knowmem_ROUGE']:.2f}%, retain={scores['retain_knowmem_ROUGE']:.2f}%")
    
    return optimal_configs

def main_scatter_plot():
    """
    Updated main scatter plot using optimal configurations.
    Left plot shows optimal for verbmem, right plot shows optimal for knowmem.
    Cleaned up privleak code - no longer prints privleak metrics.
    """
    # Find optimal configurations
    print("Finding optimal configurations...")
    optimal_configs = find_optimal_configurations()
    
    data = []
    
    # Load Target and Retrain baselines
    retrain_target = ["Target", "Retrain"]
    for name in retrain_target:
        info = json.load(open(f"saves/eval/muse_main/muse_{name.lower()}/MUSE_SUMMARY.json"))
        info["name"] = name
        data.append(info)

    # Load gradient-based methods
    gradient_methods = ["GradDiff", "NPO", "SimNPO"]
    for name in gradient_methods:
        info = json.load(open(f"saves/unlearn/muse/muse_Llama-2-7b-hf_News_{name}/evals/MUSE_SUMMARY.json"))
        info["name"] = name
        data.append(info)

    # Load DD methods using optimal configurations for verbmem (left plot)
    model_sizes = ["1.3b"] #["1.3b", "Trigram"]

    # Prepare separate data for left (verbmem) and right (knowmem) plots
    data_verbmem = data.copy()
    data_knowmem = data.copy()
    
    for model_size in model_sizes:
        if model_size in optimal_configs:
            # For verbmem plot - use verbmem optimal configurations
            if 'alpha_verbmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['alpha_verbmem']['folder']
                name = "Linear DD"#f"{model_size} Linear DD" if model_size != "1.3b" else "LLaMA Linear DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    data_verbmem.append(info)
                    print(f"Loaded {name} (verbmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")
            
            if 'topk_verbmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['topk_verbmem']['folder']
                name = "Rank DD" #f"{model_size} Rank DD" if model_size != "1.3b" else "LLaMA Rank DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    data_verbmem.append(info)
                    print(f"Loaded {name} (verbmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")
            
            # For knowmem plot - use knowmem optimal configurations
            if 'alpha_knowmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['alpha_knowmem']['folder']
                name = "Linear DD" #f"{model_size} Linear DD" if model_size != "1.3b" else "LLaMA Linear DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    data_knowmem.append(info)
                    print(f"Loaded {name} (knowmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")
            
            if 'topk_knowmem' in optimal_configs[model_size]:
                folder = optimal_configs[model_size]['topk_knowmem']['folder']
                name = "Rank DD" #f"{model_size} Rank DD" if model_size != "1.3b" else "LLaMA Rank DD"
                try:
                    info = json.load(open(f"saves/eval/{folder}/MUSE_SUMMARY.json"))
                    info["name"] = name
                    data_knowmem.append(info)
                    print(f"Loaded {name} (knowmem optimal) from {folder}")
                except FileNotFoundError:
                    print(f"Warning: Could not find {folder}")

    if not data_verbmem:
        print("No data found. Please check file paths.")
        return

    df_verbmem = pd.DataFrame(data_verbmem)
    df_verbmem["forget_verbmem_ROUGE"] *= 100
    df_verbmem["forget_knowmem_ROUGE"] *= 100
    df_verbmem["retain_knowmem_ROUGE"] *= 100
    
    df_knowmem = pd.DataFrame(data_knowmem)
    df_knowmem["forget_verbmem_ROUGE"] *= 100
    df_knowmem["forget_knowmem_ROUGE"] *= 100
    df_knowmem["retain_knowmem_ROUGE"] *= 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    s = 250

    # Left subplot: Verbatim Memorization (using verbmem optimal)
    sns.scatterplot(
        data=df_verbmem,
        x="forget_verbmem_ROUGE",
        y="retain_knowmem_ROUGE",
        hue="name",
        style="name",
        palette=palette,
        markers=markers,
        s=s,
        ax=axes[0],
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )
    axes[0].set_xlabel("Verbatim Memorization of Forget Set")
    axes[0].set_ylabel("Utility on Retain Set")

    # Right subplot: Q&A Knowledge Forget (using knowmem optimal)
    right_ax = axes[1]
    sns.scatterplot(
        data=df_knowmem,
        x="forget_knowmem_ROUGE",
        y="retain_knowmem_ROUGE",
        hue="name",
        style="name",
        palette=palette,
        markers=markers,
        s=s,
        ax=right_ax,
        edgecolor="black",
        linewidth=0.5,
        legend="full",
    )
    right_ax.set_xlabel("Q&A Knowledge of Forget Set")
    right_ax.set_ylabel(None)

    axes[0].set_xlim(10, 60)
    axes[0].set_ylim(40, 60)
    axes[1].set_xlim(20, 70)
    axes[0].set_yticks([40, 45, 50, 55, 60])

    # Build legend - group by model size and method type
    handles, labels = right_ax.get_legend_handles_labels()
    if labels and labels[0].lower() == "name":
        handles, labels = handles[1:], labels[1:]

    # Custom legend order: Baselines, Gradient methods, then DD methods by size
    legend_order = [
        "Target", "Retrain", "GradDiff", "NPO", "SimNPO",
        "Linear DD", "Rank DD"]
    #     "Trigram Linear DD", "Trigram Rank DD"
    # ]
    
    # Reorder handles and labels
    ordered_handles = []
    ordered_labels = []
    for desired_label in legend_order:
        if desired_label in labels:
            idx = labels.index(desired_label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    # Place legend at the top
    # First row: 5 items
    legend1 = fig.legend(
        ordered_handles, ordered_labels,
        loc="upper center",
        ncol=7,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.6,
        borderaxespad=0.5,
        bbox_to_anchor=(0.5, 1.05),
    )

    # Add both legends to the figure
    fig.add_artist(legend1)

    # Remove the legend from the right subplot
    leg = right_ax.get_legend()
    if leg:
        leg.remove()

    plt.tight_layout()
    plt.savefig("muse_scatter_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("muse_scatter_plot.pdf", dpi=600, bbox_inches="tight")

def plot_muse_curve():
    """
    Create scatter plots showing performance curves for MUSE models with different alpha values
    and topk values. Shows separate optimal points for verbmem and knowmem.
    Uses color gradients based on hyperparameter values.
    """
    
    # Define base colors for different model sizes
    base_colors = {
        "Target": "#1f77b4",
        "Retrain": "#ff7f0e", 
        "1.3B": "Reds",
        "2.7B": "Blues", 
        "Trigram": "Greens",
    }
    
    curve_markers = {
        "baseline": "s", 
        "alpha": "o",
        "topk": "X",
    }

    sizes = {
        "baseline": 180,
        "alpha": 100,
        "topk": 100,
    }
    
    def get_color_from_gradient(cmap_name, value, value_range):
        """Get color from matplotlib colormap based on normalized value"""
        rank = value_range.index(value)
        color_intensity = 0.5 + rank/len(value_range) * 0.4
        cmap = plt.cm.get_cmap(cmap_name)
        #print(f"Value: {value}, Rank: {rank}, Intensity: {color_intensity}, Color: {cmap(color_intensity)}, Len: {len(value_range)}")
        return cmap(color_intensity)
    
    data = []
    
    # Load Target and Retrain data
    retrain_target = ["Target", "Retrain"]
    for name in retrain_target:
        try:
            info = json.load(open(f"saves/eval/muse_main/muse_{name.lower()}/MUSE_SUMMARY.json"))
            info["name"] = name
            info["alpha"] = None
            info["topk"] = None
            info["model_size"] = "baseline"
            info["method"] = "baseline"
            info["is_optimal_verbmem"] = False
            info["is_optimal_knowmem"] = False
            info["color"] = base_colors[name]
            data.append(info)
        except FileNotFoundError:
            continue
    
    # Load data for all model sizes
    model_sizes = ["1.3b", "2.7b", "Trigram",]  #"7b"
    
    for model_size in model_sizes:

        if model_size == "Trigram":
            alpha_values = alpha_values_trigram
            topk_values = topk_values_trigram
        else:
            alpha_values = alpha_values_model
            topk_values = topk_values_model


        model_name = f"{model_size.replace('b', 'B')}"
        cmap_name = base_colors[model_name]
        
        # Load alpha-based models
        for alpha in alpha_values:
            folder_name = f"muse_main/muse-{model_size}-alpha-{alpha}"
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                info["name"] = model_name
                info["alpha"] = alpha
                info["topk"] = None
                info["model_size"] = model_size
                info["method"] = "alpha"

                info["color"] = get_color_from_gradient(cmap_name, alpha, alpha_values)
                
                data.append(info)
            except FileNotFoundError:
                continue
        
        # Load topk-based models
        for topk in topk_values:
            folder_name = f"muse_main/muse-{model_size}-topk-{topk}"
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                info["name"] = model_name
                info["alpha"] = None
                info["topk"] = topk
                info["model_size"] = model_size
                info["method"] = "topk"
                
                # Color based on topk value
                info["color"] = get_color_from_gradient(cmap_name, topk, topk_values)
                
                data.append(info)
            except FileNotFoundError:
                continue
    
    if not data:
        print("No data found. Please check that saves/eval/ directory exists and contains MUSE model folders.")
        return
    
    df = pd.DataFrame(data)
    
    # Convert to percentages
    df["forget_verbmem_ROUGE"] *= 100
    df["forget_knowmem_ROUGE"] *= 100
    df["retain_knowmem_ROUGE"] *= 100
    df["size"] = df["method"].map(sizes)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    # Plot each point individually to control colors
    for idx, row in df.iterrows():
        for ax_idx, x_col in enumerate(["forget_verbmem_ROUGE", "forget_knowmem_ROUGE"]):
            axes[ax_idx].scatter(
                row[x_col], 
                row["retain_knowmem_ROUGE"],
                c=[row["color"]], 
                marker=curve_markers[row["method"]],
                s=sizes[row["method"]],
                edgecolor="black",
                linewidth=0.5,
                alpha=1
            )
    
    axes[0].set_xlabel("Verbatim Memorization of Forget Set")
    axes[0].set_ylabel("Utility on Retain Set")
    #axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Q&A Knowledge of Forget Set")
    axes[1].set_ylabel("")  # shared y-axis
    #axes[1].grid(True, alpha=0.3)
    
    # Set consistent axis limits
    axes[0].set_xlim(10, 60)
    axes[0].set_ylim(20, 60)
    axes[1].set_xlim(20, 70)
    axes[0].set_yticks([30, 40, 50, 60])
    
    # Build legend manually with lightest colors
    legend_handles = []
    legend_labels = []
    
    # Define the order of legend items
    legend_items = ["Target", "Retrain", "Trigram", "1.3B", "2.7B", "alpha", "topk"] #"DD 7b",
    
    for model_name in legend_items:
        if model_name in ["Target", "Retrain"]:
            color = base_colors[model_name]
            marker = curve_markers["baseline"]
        elif model_name in ["Trigram", "1.3B", "2.7B"]:
            # Use lightest color from the colormap (0.3 intensity)
            cmap_name = base_colors[model_name]
            cmap = plt.cm.get_cmap(cmap_name)
            color = cmap(0.75)  # Median Color
            marker = 's'  # Default marker for model representation
        elif model_name == "alpha":
            color = 'gray'
            marker = curve_markers["alpha"]
            model_name = "Linear DD"
        elif model_name == "topk":
            color = 'gray'
            marker = curve_markers["topk"]
            model_name = "Rank DD"
        else:
            color = 'gray'
            marker = 's'
        
        handle = plt.Line2D([0], [0], marker=marker, color='w', 
                           markerfacecolor=color, markersize=10,
                           markeredgecolor='black', markeredgewidth=0.5)
        legend_handles.append(handle)
        legend_labels.append(model_name)

    # Place legend at the top
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        ncol=10,  
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.6,
        borderaxespad=0.5,
        bbox_to_anchor=(0.5, 1.05),
    )
    
    plt.tight_layout()
    plt.savefig("muse_alpha_topk_curves.png", dpi=600, bbox_inches="tight")
    plt.savefig("muse_alpha_topk_curves.pdf", dpi=600, bbox_inches="tight")

def model_scaling_plot():
    """
    Create a line plot showing how model scaling affects performance for different methods.
    Now shows two lines: best method for Q&A Questions and best method for Verbatim Memorization.
    Combines both alpha and topk methods, selecting the best performing option for each.
    Uses euclidean distance only.
    """
    # Find optimal configurations
    print(f"Finding optimal configurations for model scaling plot...")
    optimal_configs = find_optimal_configurations()

    # Load Target and Retrain baselines
    retrain_target = ["Target", "Retrain"]
    baseline_scores = {}
    
    for name in retrain_target:
        try:
            info = json.load(open(f"saves/eval/muse_main/muse_{name.lower()}/MUSE_SUMMARY.json"))
            baseline_scores[name] = {
                'forget_verbmem_ROUGE': info['forget_verbmem_ROUGE'] * 100,
                'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
            }
        except FileNotFoundError:
            print(f"Warning: Could not find baseline data for {name}")
            continue
    
    # If we don't have retrain baseline or target, we can't compute distances
    if 'Retrain' not in baseline_scores or 'Target' not in baseline_scores:
        print("Error: Retrain or Target baseline not found. Cannot compute normalized distances.")
        return
    
    retrain_scores = baseline_scores['Retrain']
    target_scores = baseline_scores['Target']
    
    # Load data for all model sizes using optimal configurations
    model_sizes = ["Trigram", "1.3b", "2.7b"] #"7b"
    model_size_x = {"Trigram": 0, "1.3b": 1.3, "2.7b": 2.7}
    
    results = []
    
    # Function to calculate distance for verbmem optimization
    def calculate_distance_verbmem(point):
        forget_verbmem_diff = point['forget_verbmem_ROUGE'] - retrain_scores['forget_verbmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (forget_verbmem_diff**2 + retain_diff**2)**0.5
    
    # Function to calculate distance for knowmem optimization
    def calculate_distance_knowmem(point):
        forget_knowmem_diff = point['forget_knowmem_ROUGE'] - retrain_scores['forget_knowmem_ROUGE']
        retain_diff = point['retain_knowmem_ROUGE'] - retrain_scores['retain_knowmem_ROUGE']
        return (forget_knowmem_diff**2 + retain_diff**2)**0.5
    
    # Calculate target distances for normalization
    target_distance_verbmem = calculate_distance_verbmem(target_scores)
    target_distance_knowmem = calculate_distance_knowmem(target_scores)
    
    # Function to normalize distances relative to target (target = 100%)
    def normalize_distance_verbmem(distance):
        return (distance / target_distance_verbmem) * 100
    
    def normalize_distance_knowmem(distance):
        return (distance / target_distance_knowmem) * 100
    
    # Prepare data for seaborn
    plot_data = []
    
    for model_size in model_sizes:
        print(f"Processing {model_size}...")
        
        x_pos = model_size_x[model_size]
        
        if model_size in optimal_configs:
            # Find best method for verbmem (memorization)
            verbmem_candidates = []
            
            if 'alpha_verbmem' in optimal_configs[model_size]:
                alpha_scores = optimal_configs[model_size]['alpha_verbmem']['scores']
                alpha_distance = calculate_distance_verbmem(alpha_scores)
                normalized_alpha_distance = normalize_distance_verbmem(alpha_distance)
                verbmem_candidates.append({
                    'method_name': 'alpha',
                    'param_value': optimal_configs[model_size]['alpha_verbmem']['value'],
                    'distance': normalized_alpha_distance
                })
                print(f"  Alpha verbmem: α={optimal_configs[model_size]['alpha_verbmem']['value']}, distance={normalized_alpha_distance:.2f}%")
            
            if 'topk_verbmem' in optimal_configs[model_size]:
                topk_scores = optimal_configs[model_size]['topk_verbmem']['scores']
                topk_distance = calculate_distance_verbmem(topk_scores)
                normalized_topk_distance = normalize_distance_verbmem(topk_distance)
                verbmem_candidates.append({
                    'method_name': 'topk',
                    'param_value': optimal_configs[model_size]['topk_verbmem']['value'],
                    'distance': normalized_topk_distance
                })
                print(f"  TopK verbmem: k={optimal_configs[model_size]['topk_verbmem']['value']}, distance={normalized_topk_distance:.2f}%")
            
            # Select best verbmem method (lowest distance)
            if verbmem_candidates:
                best_verbmem = min(verbmem_candidates, key=lambda x: x['distance'])
                plot_data.append({
                    'model_size': model_size,
                    'x_pos': x_pos,
                    'distance': best_verbmem['distance'],
                    'method': 'Verbatim Memorization',
                    'best_method': best_verbmem['method_name'],
                    'best_param': best_verbmem['param_value']
                })
                print(f"  Best verbmem: {best_verbmem['method_name']} with {best_verbmem['param_value']}, distance={best_verbmem['distance']:.2f}%")
            
            # Find best method for knowmem (Q&A Questions)
            knowmem_candidates = []
            
            if 'alpha_knowmem' in optimal_configs[model_size]:
                alpha_scores = optimal_configs[model_size]['alpha_knowmem']['scores']
                alpha_distance = calculate_distance_knowmem(alpha_scores)
                normalized_alpha_distance = normalize_distance_knowmem(alpha_distance)
                knowmem_candidates.append({
                    'method_name': 'alpha',
                    'param_value': optimal_configs[model_size]['alpha_knowmem']['value'],
                    'distance': normalized_alpha_distance
                })
                print(f"  Alpha knowmem: α={optimal_configs[model_size]['alpha_knowmem']['value']}, distance={normalized_alpha_distance:.2f}%")
            
            if 'topk_knowmem' in optimal_configs[model_size]:
                topk_scores = optimal_configs[model_size]['topk_knowmem']['scores']
                topk_distance = calculate_distance_knowmem(topk_scores)
                normalized_topk_distance = normalize_distance_knowmem(topk_distance)
                knowmem_candidates.append({
                    'method_name': 'topk',
                    'param_value': optimal_configs[model_size]['topk_knowmem']['value'],
                    'distance': normalized_topk_distance
                })
                print(f"  TopK knowmem: k={optimal_configs[model_size]['topk_knowmem']['value']}, distance={normalized_topk_distance:.2f}%")
            
            # Select best knowmem method (lowest distance)
            if knowmem_candidates:
                best_knowmem = min(knowmem_candidates, key=lambda x: x['distance'])
                plot_data.append({
                    'model_size': model_size,
                    'x_pos': x_pos,
                    'distance': best_knowmem['distance'],
                    'method': 'Q&A Questions',
                    'best_method': best_knowmem['method_name'],
                    'best_param': best_knowmem['param_value']
                })
                print(f"  Best knowmem: {best_knowmem['method_name']} with {best_knowmem['param_value']}, distance={best_knowmem['distance']:.2f}%")
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    # Create the plot using seaborn
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    
    # Use seaborn lineplot with markers
    sns.lineplot(data=df, x='x_pos', y='distance', hue='method', 
                marker='s', markersize=8, linewidth=2, ax=ax)

    # Customize the plot
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Distance")
    ax.set_title("Model Scaling", fontsize=14)

    # Add invisible top axis to match right plot spacing
    ax_top = ax.twiny()
    ax_top.set_xticks([])  # No ticks
    ax_top.set_xlabel("")  # No label  
    #ax_top.tick_params(top=False, labeltop=False)  # Hide ticks and labels
    
    # Set x-axis labels
    ax.set_xticks([0, 1.3, 2.7]) #, 7.0
    ax.set_xticklabels(['Trigram', '1.3B', '2.7B'])#'7b'
    
    ax.set_xlim(-0.3, 3.0)
    
    # Set y-axis limits to show 0% to slightly above 100%
    ax.set_ylim(20, 60)
    ax.invert_yaxis()  
    
    # Add y-axis ticks at 0%, 25%, 50%, 75%, and 100%
    ax.set_yticks([25, 35, 45, 55])
    ax.set_yticklabels(['25%', '35%', '45%', '55%'])
    
    # Position legend
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f"model_scaling_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"model_scaling_plot.pdf", dpi=600, bbox_inches="tight")

def scal_sust():
    fig, ax = plt.subplots(2, 2, figsize=(10, 4), sharey='row', sharex='col')
    plt.tight_layout()
    plt.savefig("muse_scal_sust.png", dpi=600, bbox_inches="tight")
    plt.savefig("muse_scal_sust.pdf", dpi=600, bbox_inches="tight")

    # Left will be Sust, Right will be Scal
    # Top will be Knowmem_R, Bottom will be Knowmem_f

    data = []
    info0 = json.load(open("saves/eval/muse_main/muse_target/MUSE_SUMMARY.json"))
    for bench in ["sust", "scal"]:
        files = [
            ("saves/eval/muse_main/muse_target/MUSE_SUMMARY.json", 0, 'Linear DD'),
            ("saves/eval/muse_main/muse_target/MUSE_SUMMARY.json", 0, 'Rank DD'),
            ("saves/eval/muse_main/muse-1.3b-alpha-0.8/MUSE_SUMMARY.json", 1, 'Linear DD'),
            ("saves/eval/muse_main/muse-1.3b-topk-1000/MUSE_SUMMARY.json", 1, 'Rank DD'),
        ]
        for file, num, method in files:
            data.append({
                'method': method,
                'bench': bench,
                'num': num,
                'forget_knowmem_ROUGE': json.load(open(file))['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': json.load(open(file))['retain_knowmem_ROUGE'] * 100
            })

        for method in ['GradDiff', 'NPO', 'SimNPO']:
            data.append({
                'method': method,
                'bench': bench,
                'num': 0,
                'forget_knowmem_ROUGE': info0['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info0['retain_knowmem_ROUGE'] * 100,
            })
            info1 = json.load(open(f"saves/unlearn/muse/muse_Llama-2-7b-hf_News_{method}/evals/MUSE_SUMMARY.json"))
            data.append({
                'method': method,
                'bench': bench,
                'num': 1,
                'forget_knowmem_ROUGE': info1['forget_knowmem_ROUGE'] * 100,
                'retain_knowmem_ROUGE': info1['retain_knowmem_ROUGE'] * 100
            })
            for num in [2, 3, 4]:
                try:
                    info = json.load(open(f"saves/unlearn/muse/muse_Llama-2-7b-hf_News_{method}_{bench}_forget_{num}/evals/MUSE_SUMMARY.json"))
                    data.append({
                        'method': method,
                        'bench': bench,
                        'num': num,
                        'forget_knowmem_ROUGE': info['forget_knowmem_ROUGE'] * 100,
                        'retain_knowmem_ROUGE': info['retain_knowmem_ROUGE'] * 100
                    })
                except FileNotFoundError:
                    pass
    
    files = [
        ("saves/eval/muse_scalsust/muse-scalsust-linear-1.3b-3/MUSE_SUMMARY.json", 2, 'Linear DD', 'scal'),
        ("saves/eval/muse_scalsust/muse-scalsust-linear-1.3b-4/MUSE_SUMMARY.json", 3, 'Linear DD', 'scal'),
        ("saves/eval/muse_scalsust/muse-scalsust-linear-1.3b-5/MUSE_SUMMARY.json", 4, 'Linear DD', 'scal'),
        ("saves/eval/muse_scalsust/muse-scalsust-rank-1.3b-3/MUSE_SUMMARY.json", 2, 'Rank DD', 'scal'),
        ("saves/eval/muse_scalsust/muse-scalsust-rank-1.3b-4/MUSE_SUMMARY.json", 3, 'Rank DD', 'scal'),
        ("saves/eval/muse_scalsust/muse-scalsust-rank-1.3b-5/MUSE_SUMMARY.json", 4, 'Rank DD', 'scal'),
        ("saves/eval/muse_scalsust/muse-scalsust-linear-1.3b-6/MUSE_SUMMARY.json", 2, 'Linear DD', 'sust'),
        ("saves/eval/muse_scalsust/muse-scalsust-linear-1.3b-7/MUSE_SUMMARY.json", 3, 'Linear DD', 'sust'),
        ("saves/eval/muse_scalsust/muse-scalsust-linear-1.3b-8/MUSE_SUMMARY.json", 4, 'Linear DD', 'sust'),
        ("saves/eval/muse_scalsust/muse-scalsust-rank-1.3b-6/MUSE_SUMMARY.json", 2, 'Rank DD', 'sust'),
        ("saves/eval/muse_scalsust/muse-scalsust-rank-1.3b-7/MUSE_SUMMARY.json", 3, 'Rank DD', 'sust'),
        ("saves/eval/muse_scalsust/muse-scalsust-rank-1.3b-8/MUSE_SUMMARY.json", 4, 'Rank DD', 'sust'),
    ]
    for file, num, method, bench in files:
        data.append({
            'method': method,
            'bench': bench,
            'num': num,
            'forget_knowmem_ROUGE': json.load(open(file))['forget_knowmem_ROUGE'] * 100,
            'retain_knowmem_ROUGE': json.load(open(file))['retain_knowmem_ROUGE'] * 100
        })

    palette['black_dot'] = 'black'

    data.append({
        'method': 'black_dot',
        'bench': 'scal',
        'num': 0,
        'forget_knowmem_ROUGE': info0['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': info0['retain_knowmem_ROUGE'] * 100,
    })

    data.append({
        'method': 'black_dot',
        'bench': 'sust',
        'num': 0,
        'forget_knowmem_ROUGE': info0['forget_knowmem_ROUGE'] * 100,
        'retain_knowmem_ROUGE': info0['retain_knowmem_ROUGE'] * 100,
    })


    data = pd.DataFrame(data)

    sns.lineplot(
        data=data[data['bench'] == 'sust'],
        x='num',
        y='retain_knowmem_ROUGE',
        hue='method',
        marker='o',
        ax=ax[0][0],
        legend=False,
        palette=palette
    )
    sns.lineplot(
        data=data[data['bench'] == 'scal'],
        x='num',
        y='retain_knowmem_ROUGE',
        hue='method',
        marker='o',
        ax=ax[0][1],
        legend=False,
        palette=palette
    )
    sns.lineplot(
        data=data[data['bench'] == 'sust'],
        x='num',
        y='forget_knowmem_ROUGE',
        hue='method',
        marker='o',
        ax=ax[1][0],
        legend=False,
        palette=palette
    )
    sns.lineplot(
        data=data[data['bench'] == 'scal'],
        x='num',
        y='forget_knowmem_ROUGE',
        hue='method',
        marker='o',
        ax=ax[1][1],
        legend='full',
        palette=palette
    )

    ax[0][0].set_title("Sustainability")
    ax[0][1].set_title("Scalability")
    ax[0][0].set_ylabel("Utility")
    ax[1][0].set_ylabel("Forget")
    ax[1][0].set_xlabel("Nth Forget Request")
    ax[1][1].set_xlabel("Size of Forget Set")

    ax[0][0].set_xticks([0, 1, 2, 3, 4])
    ax[0][1].set_xticks([0, 1, 2, 3, 4])
    ax[1][0].set_xticks([0, 1, 2, 3, 4])
    ax[1][1].set_xticks([0, 1, 2, 3, 4])

    ax[1][0].set_xticklabels(['0th', '1st', '2nd', '3rd', '4th'])
    ax[1][1].set_xticklabels(['0.0M', '0.8M', '1.7M', '2.5M', '3.3M'])

    #Put the legend on the bottom
    handles, labels = ax[1][1].get_legend_handles_labels()
    handles = handles[:-1]
    labels = labels[:-1]

    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.6,
        borderaxespad=0.5,
        bbox_to_anchor=(0.5, 0.03),
    )

    #hide the legend on the right bottom plot
    ax[1][1].get_legend().remove()

    plt.tight_layout()
    plt.savefig("muse_scal_sust.png", dpi=600, bbox_inches="tight")
    plt.savefig("muse_scal_sust.pdf", dpi=600, bbox_inches="tight")

def privleak_plot():
    """
    Create a plot showing privleak vs alpha (bottom axis) and privleak vs topk (top twinx axis on log scale).
    Uses only 1.3b model size. Alpha values from 0 to 3 with 0.5 spacing. Uses 'target' value for alpha=0.
    """
    
    # Get optimal configurations to mark optimal points
    optimal_configs = find_optimal_configurations()
    
    alpha_data = []
    topk_data = []
    
    # Load target baseline for alpha=0
    target_info = json.load(open("saves/eval/muse_main/muse_target/MUSE_SUMMARY.json"))
    target_privleak = target_info.get('privleak', None)
    print(f"Target privleak value: {target_privleak}")

    retrain_info = json.load(open("saves/eval/muse_main/muse_retrain/MUSE_SUMMARY.json"))
    retrain_privleak = retrain_info.get('privleak', None)
    print(f"Retrain privleak value: {retrain_privleak}")
    
    # Model sizes to process (only 1.3b)
    model_sizes = ["1.3b"]
    
    # Alpha values from 0 to 3 with 0.5 spacing
    alpha_values_full = [round(x * 0.5, 1) for x in range(0, 7)] 
    
    for model_size in model_sizes:
        print(f"Loading privleak data for {model_size}...")
        
        # Add target value for alpha=0 if available
        if target_privleak is not None:
            alpha_data.append({
                'model_size': model_size,
                'alpha': 0.0,
                'privleak': target_privleak
            })
        
        alpha_values = alpha_values_full[1:]  # Skip alpha=0 since we use target
        
        # Load alpha-based models for this model size
        for alpha in alpha_values:
            folder_name = f"muse_main/muse-{model_size}-alpha-{alpha}"
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                privleak_value = info.get('privleak', None)
                
                if privleak_value is not None:
                    alpha_data.append({
                        'model_size': model_size,
                        'alpha': alpha,
                        'privleak': privleak_value
                    })
                    
            except FileNotFoundError:
                print(f"Warning: Could not find {folder_name}")
                continue
        
        # Load topk-based models for this model size
        topk_values = topk_values_model
            
        for topk in topk_values:
            folder_name = f"muse_main/muse-{model_size}-topk-{topk}"
            try:
                info = json.load(open(f"saves/eval/{folder_name}/MUSE_SUMMARY.json"))
                privleak_value = info.get('privleak', None)
                
                if privleak_value is not None:
                    topk_data.append({
                        'model_size': model_size,
                        'topk': topk,
                        'privleak': privleak_value
                    })
                    
            except FileNotFoundError:
                print(f"Warning: Could not find {folder_name}")
                continue
    
    if not alpha_data and not topk_data:
        print("No privleak data found. Please check that the MUSE_SUMMARY.json files contain 'privleak' values.")
        return
    
    # Convert to DataFrames
    df_alpha = pd.DataFrame(alpha_data)
    df_topk = pd.DataFrame(topk_data)
    
    # Create the plot with twinx
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))
    
    # Use consistent colors from the palette
    linear_color = palette['Linear DD']
    rank_color = palette['Rank DD']
    
    # Bottom plot: Alpha vs Privleak (Linear DD)
    if not df_alpha.empty:
        ax1.plot(df_alpha['alpha'], df_alpha['privleak'], 
                marker='o', markersize=8, linewidth=2, 
                color=linear_color, 
                label="Linear DD")
    
    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Privacy Leakage")
    
    # Create twin axis for topk
    ax2 = ax1.twiny()
    
    # Top plot: TopK vs Privleak on log scale (Rank DD)
    if not df_topk.empty:
        ax2.plot(df_topk['topk'], df_topk['privleak'], 
                marker='X', markersize=8, linewidth=2, linestyle='--',
                color=rank_color, 
                label="Rank DD")
    
    ax2.set_xlabel("Top-k")
    ax2.set_xscale('log')
    
    # Add retrain baseline
    if retrain_privleak is not None:
        ax1.axhline(retrain_privleak, color='black', linestyle='--', alpha=1, label='Retrain')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Order: Linear DD, Rank DD, then retrain
    ordered_lines = []
    ordered_labels = []
    
    # Add Linear DD
    if "Linear DD" in labels1:
        idx = labels1.index("Linear DD")
        ordered_lines.append(lines1[idx])
        ordered_labels.append("Linear DD")
    
    # Add Rank DD
    if "Rank DD" in labels2:
        idx = labels2.index("Rank DD")
        ordered_lines.append(lines2[idx])
        ordered_labels.append("Rank DD")
    
    # Add retrain if present
    if 'Retrain' in labels1:
        idx = labels1.index('Retrain')
        ordered_lines.append(lines1[idx])
        ordered_labels.append('Retrain')
    
    ax1.legend(ordered_lines, ordered_labels, loc='lower right')
    ax1.set_yticks([0])
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("privleak_plot.png", dpi=600, bbox_inches="tight")
    plt.savefig("privleak_plot.pdf", dpi=600, bbox_inches="tight")
    
    print(f"Privleak plot saved as privleak_plot.png and privleak_plot.pdf")
    print(f"Processed {len(df_alpha)} alpha data points and {len(df_topk)} topk data points")


# Main execution
if __name__ == "__main__":
    print("Running all plotting functions with separate verbmem/knowmem optimization...\n")
    
    # Run all plotting functions
    main_scatter_plot()
    plot_muse_curve() 
    model_scaling_plot()
    scal_sust()
    privleak_plot()