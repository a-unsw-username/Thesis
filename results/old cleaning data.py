import json

#model = "hubert"
model = "w2v2"
#model = "mms"

filepath_w2v2 = "D:/Users/andrew/uni_work/thesis/code/results/wav2vec2large_kenlm/"
filepath_hubert = "D:/Users/andrew/uni_work/thesis/code/results/hubertlarge_kenlm/"
filepath_mms = "D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/"

filename = "finetune_CU_20250630_finetuned_result.json"

if model == "mms":
    filepath = filepath_mms
elif model == "w2v2":
    filepath = filepath_w2v2
else:
    filepath = filepath_hubert

file = filepath + filename

#------------------------------------------------------
def clean_text(text):
    """Clean text by replacing <unk> and unicode characters with spaces"""
    if isinstance(text, str):
        return text.replace('<unk>', ' ').replace('\u2047', ' ').replace('⁇', ' ')
    return text

def clean_dict_values(data_dict):
    """Clean all string values in a dictionary"""
    if isinstance(data_dict, dict):
        for key, value in data_dict.items():
            if isinstance(value, str):
                data_dict[key] = clean_text(value)
            elif isinstance(value, dict):
                clean_dict_values(value)
    return data_dict

# Load the JSON lines file
cleaned_data = []
line_count = 0
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        line_count += 1
        entry = json.loads(line)
        
        # Clean text in 'pred_str' dictionary
        if 'pred_str' in entry and isinstance(entry['pred_str'], dict):
            clean_dict_values(entry['pred_str'])
        
        # Clean text in 'text' dictionary
        if 'text' in entry and isinstance(entry['text'], dict):
            clean_dict_values(entry['text'])
        
        # Clean text in 'n_best' dictionary (special handling for nested structure)
        if 'n_best' in entry and isinstance(entry['n_best'], dict):
            for nbest_key, nbest_list in entry['n_best'].items():
                if isinstance(nbest_list, list):
                    # Each n_best item is a list of dictionaries
                    for item in nbest_list:
                        if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                            old_text = item['text']
                            item['text'] = clean_text(item['text'])
                            if old_text != item['text']:
                                print(f"Cleaned n_best[{nbest_key}] text: {repr(old_text)} -> {repr(item['text'])}")
        
        cleaned_data.append(entry)

print(f"Total lines processed: {line_count}")
print("Cleaning completed!")

# Save cleaned data back
new_file = filepath + "cleaned_" + filename
with open(new_file, "w", encoding="utf-8") as f:
    for item in cleaned_data:
        json.dump(item, f)
        f.write("\n")

print(f"Cleaned data saved to: {new_file}")