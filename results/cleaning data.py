import json

model = "mms"


filepath_w2v2 = "D:/Users/andrew/uni_work/thesis/code/results/wav2vec2large_kenlm/"
filepath_hubert = "D:/Users/andrew/uni_work/thesis/code/results/hubertlarge_kenlm/"
filepath_mms = "D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/"

filename = "finetune_CU_20250703_finetuned_result.json"

if model == "mms":
    filepath = filepath_mms
elif model == "w2v2":
    filepath = filepath_w2v2
else:
    filepath = filepath_hubert

file = filepath + filename

# Load the JSON lines file
cleaned_data = []
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)

        # Replace <unk> and \u2047 with a space in relevant fields
        for key in ['pred_str', 'text']:
            if key in entry:
                entry[key] = entry[key].replace('<unk>', ' ').replace('\u2047', ' ')

        # Also clean 'text' fields in 'n_best'
        if 'n_best' in entry:
            for item in entry['n_best']:
                item['text'] = item['text'].replace('<unk>', ' ').replace('\u2047', ' ')

        cleaned_data.append(entry)

# Optionally: save cleaned data back
new_file = filepath + "cleaned_" + filename
with open(new_file, "w", encoding="utf-8") as f:
    for item in cleaned_data:
        json.dump(item, f)
        f.write("\n")