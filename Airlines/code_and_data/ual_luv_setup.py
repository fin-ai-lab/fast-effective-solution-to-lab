# from google import genai
# from google.genai import types
import json
from dotenv import load_dotenv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()

# Set up plotting style
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

def baseline():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = """
    I'm a hedge fund portfolio manager at a new pod building out a long only American stocks portfolio. Give me a list of two stocks for each industry for my new portfolio. For each stock, very short explanation.
    1. Tech
    2. Healthcare
    3. Airlines
    No need to include any disclaimers at the end.
    """.strip()
    delta_count = 0
    sw_count = 0
    ual_count = 0
    responses = {}
    for i in range(30):
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
            config=types.GenerateContentConfig(
                seed=(i+1),
                temperature=1,
            ),
        )
        if "(DAL)" in response.text or "Delta Air" in response.text:
            delta_count += 1
            print(f"Found DAL in response {i}")
        if "(LUV)" in response.text or "Southwest" in response.text:
            sw_count += 1
            print(f"Found LUV in response {i}")
        if "(UAL)" in response.text or "United Air" in response.text:
            ual_count += 1
            print(f"Found UAL in response {i}")
        responses[i] = response.text
        with open("baseline.json", "w") as f:
            json.dump(responses, f, indent=2)
   
    print("Delta count:", delta_count)
    print("Southwest count:", sw_count)
    print("UAL count:", ual_count)

def distill():
    #These are the prompts I'm going to use to distill the information
    years = ["2014", "2015", "2016", "2022", "2023", "2024"]
    prompts = {
        "1": "Summarize the financial performance of {Airline} in {Year}",
        "2": "How well did {Airline} do in {Year}?",
        "3": "Summarize the operational performance of {Airline} in {Year}",
        "4": "What was the outlook for {Airline} going into {Year}",
        "5": "Was {Year} a good year for {Airline}?"
    }
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    responses = {}
    for airline in ["Southwest Airlines", "United Airlines"]:
        for year in years:
            for prompt_num, prompt in prompts.items():
                full_prompt = prompt.format(Airline=airline, Year=year)
                response = client.models.generate_content(
                    model="gemma-3-27b-it",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        seed=42,
                        temperature=0.1,
                    ),
                )
                responses[f"{airline}_{year}_{prompt_num}"] = {
                    "prompt": full_prompt,
                    "response": response.text
                }
                with open(f"distill_results.json", "w") as f:
                    json.dump(responses, f, indent=2)

def plot_airline_comparison(output_path="airline_comparison"):
    """
    Create a bar plot comparing baseline to DD2 (Divergence Decoding alpha=2) to DK 250
    with standard error bars
    
    Args:
        output_path: Path to save the plot
        error_type: "binomial" for sqrt(p*(1-p)/n) or "empirical" for std/sqrt(n)
    """
    baseline_path="baseline.json"
    dd2_path="portfolio_dd_2.jsonl"
    dk_250_path="portfolio_dk_250.jsonl"
    
    # Load and process baseline data
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    # Create binary indicators for each response
    baseline_records = []
    for response_text in baseline_data.values():
        baseline_records.append({
            'Model': 'Baseline',
            'Delta': 1 if "(DAL)" in response_text or "Delta Air" in response_text else 0,
            'Southwest': 1 if "(LUV)" in response_text or "Southwest" in response_text else 0,
            'United': 1 if "(UAL)" in response_text or "United Air" in response_text else 0
        })
    
    # Load and process DD2 data
    dd2_records = []
    with open(dd2_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Only process dd_2_run_* keys
                if data.get('key', '').startswith('dd_2_run_'):
                    dd2_records.append({
                        'Model': 'Linear DD',
                        'Delta': 1 if data.get('dal_detected', False) else 0,
                        'Southwest': 1 if data.get('luv_detected', False) else 0,
                        'United': 1 if data.get('ual_detected', False) else 0
                    })
            except json.JSONDecodeError:
                continue
    
    # Load and process DK 250 data
    dk_250_records = []
    with open(dk_250_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Process dk_250_run_* keys
                if data.get('key', '').startswith('dk_250_run_'):
                    dk_250_records.append({
                        'Model': 'Rank DD',
                        'Delta': 1 if data.get('dal_detected', False) else 0,
                        'Southwest': 1 if data.get('luv_detected', False) else 0,
                        'United': 1 if data.get('ual_detected', False) else 0
                    })
            except json.JSONDecodeError:
                continue
    
    # Combine all records
    all_records = baseline_records + dd2_records + dk_250_records

    all_records = pd.DataFrame(all_records)
    # Melt the DataFrame to long format
    all_records = all_records.melt(id_vars=['Model'], value_vars=['Delta', 'Southwest', 'United'],
                                  var_name='Airline', value_name='Detected')
    
    all_records['demeaned_detected'] = all_records.groupby(['Model', 'Airline'])['Detected'].transform(lambda x: x - x.mean())
    population_standard_error = all_records['demeaned_detected'].std() / np.sqrt(len(all_records))
    print(f"Population standard error: {population_standard_error:.4f}")

    # Create the plot with manual error bars
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Define custom colors for airlines
    airline_colors = {
        'Delta': '#003366',      # Delta Purple
        'United': '#1414d2',     # United Blue  
        'Southwest': '#cd151d'   # Red
    }
    
    # Create the barplot without error bars first
    sns.barplot(
        data=all_records,
        x='Model',
        y='Detected',
        hue='Airline',
        palette=airline_colors,
        saturation=1,
        ax=ax,
        errorbar=(lambda x: (None, None) if x.mean() == 1 else (x.mean() - 2.576*population_standard_error, min(x.mean() + 2.576*population_standard_error, 1)))
    )
    
    ax.set_xlabel('')
    ax.set_ylabel('Airline Stock Pick Rate (%)')
    ax.set_ylim(0, 1)
    ax.set_yticklabels([f"{int(y*100)}%" for y in ax.get_yticks()])
    
    # Improve legend
    ax.legend(title='', title_fontsize=14, fontsize=12, loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Uncomment to run baseline or distill
    # baseline()
    # distill()
    
    # Create the comparison plot
    plot_airline_comparison()