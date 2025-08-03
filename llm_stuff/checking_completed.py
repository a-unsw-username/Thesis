import json
import pandas as pd
import ollama
from jiwer import wer
from evaluate import load



results = "D:/Users/andrew/uni_work/thesis/code/results/hubertlarge_kenlm/cleaned_finetune_CU_20250702_finetuned_result_llama"
#results = "D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/kenlmresults"
results_fp = results + ".json"

# Read the JSON file
def extract_json_to_dataframe(filename):
    """
    Extract JSON data into a pandas DataFrame with columns:
    - n_best: list of dictionaries with score and text
    - pred_str: predicted string
    - text: original text
    """
    data = []
    
    # Read the file line by line since it's JSONL format (one JSON object per line)
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                json_obj = json.loads(line.strip())
                data.append(json_obj)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df


results_df = extract_json_to_dataframe(results_fp)

#------------------------------------checking the data------------------------------------
# pd.set_option('display.max_columns', None)    # Show all columns
# pd.set_option('display.max_rows', None)       # Show all rows
# pd.set_option('display.width', None)          # Don't wrap columns
# pd.set_option('display.max_colwidth', 250)    # Set max column width
# pd.set_option('display.expand_frame_repr', False)  # Don't break into multiple lines
#-----------------------------------------------------------------------------------------


# Check the shape and structure
print(f"DataFrame shape: {results_df.shape}")
print(f"Columns: {results_df.columns.tolist()}")

# View first few rows
print(results_df.head())

# Check data types
print(results_df.dtypes)


results_new_fp = results + "_checking.json"
results_df.to_json(results_new_fp, orient="records", lines=True)
print("Saved results to:", results_new_fp)