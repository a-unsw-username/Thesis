#from ollama import chat, ChatResponse
import json
import pandas as pd
import ollama
import jiwer
from jiwer import wer
from evaluate import load

results = "D:/Users/andrew/uni_work/thesis/code/results/wav2vec2large_kenlm/cleaned_finetune_CU_20250630_finetuned_result"
#"D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/cleaned_finetune_CU_20250703_finetuned_result"
results_fp = results + ".json"
#results = "D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/kenlmresults"
#results_fp = results + ".json"

# --- tried prompt
# You are an expert in transcribing speech. Provided are the top transcription candidates with their relative logarithm confidence scores. This data has been provided by the CU speech corpus and transcribed by a trained automatic speech recognition model, you are to attempt to decrease the word error rate of these transcriptions. Select what you believe to be the most likely correct transcription (a higher value score indicates a more likely transcription) or provide a corrected version if needed - some portions of text are missing spaces between words, include spaces to separate words at your discretion. Return only the corrected text without explanations; if you believe that there may be any other issue with the score or associated text and thus do not have a definite transcription to respond with, simply provide the n_best text with the greatest score - do not provide any other text except for the transcription.
# You're seeing the results from an Automatic Speech Recognition (ASR) system. As ASR outputs can have errors, multiple candidates are provided. Please choose the one you believe is most likely correct. This text has some words connected without spaces. Correct by adding spaces in the proper location. Provide as text the corrected transcription, do not add explanations.
# You are an expert in interpreting acoustic model and language model scores. Interpret the results to determine the most probable value based upon a logrithmic score. Negative values are valid. Only suggest the most probable value and no other output. Some text include words with no spaces. Correct these and provide the transcription.
# You are an expert in interpreting acoustic model and language model scores. Interpret the results to determine the most probable text based upon the logrithmic score. Only suggest the most probable text; just output your corrected version without formatting. Some text includes words without spaces. Correct these and provide the transcription without explanation.
# You are an assistant that helps improve speech recognition predictions. Given a list of candidate transcriptions (n_best), each with a score and a predicted text, along with the reference ground truth (text) and word error rate (wer), select the best candidate that most accurately matches the reference text, correcting minor formatting errors (e.g., missing or extra spaces, punctuation, casing). If the correct transcription is not in the list, indicate that. Format your output as: Best Match: "<chosen_text>" Examples: Example 1: Reference: "T FLUENT"  n_best: ["score": -1.42, "text": "T FLUENTE", "score": -2.28, "text": "TFLUENTE", "score": -6.77, "text": "FLUENTE"]  Output:1  Best Match: "T FLUENTE"    Example 2:    Reference: "NEWSLINK"      n_best: ["score": -0.08, "text": "NEWSLINK", "score": -4.89, "text": "NEWSLINKP", "score": -8.12, "text": "NEWSLINP"]      Output:    Best Match: "NEWSLINK"
# You are an assistant that helps improve speech recognition predictions. Given a list of candidate transcriptions (n_best), each with a score and a predicted text, along with the reference ground truth (text) and word error rate (wer), select the best candidate that most accurately matches the reference text, correcting minor formatting errors (e.g., missing or extra spaces, punctuation, casing). If the correct transcription is not in the list, indicate that. Format your output as: Best Match: "<chosen_text>" Examples: Example 1: Reference: "T FLUENT"  n_best: ["score": -1.42, "text": "T FLUENTE", "score": -2.28, "text": "TFLUENTE", "score": -6.77, "text": "FLUENTE"]  Output:1   "T FLUENTE"    Example 2:    Reference: "NEWSLINK"      n_best: ["score": -0.08, "text": "NEWSLINK", "score": -4.89, "text": "NEWSLINKP", "score": -8.12, "text": "NEWSLINP"]      Output:    "NEWSLINK"
# Your an expert in validating probable values. Your provided with one or more preditions in value pairs of 'score' and 'text' within a single line json formatted file, and also a pred_str: <value> and text: <value> pair. Of the value pairs select the text from the pair with the highest score
# Of the following json pair values select the value with the highest positive score (or smallest negative value) and provide the associated text value. Neglect all other information in the line. There maybe only one pair of values in which case preovide that value
# The following speech recognition candidates are plausible texts with their associated relative probabilities as scores; some texts when generated failed to incorporate spaces. Select the most probable text and at your discretion separate words with spaces.
# --- system role
# "You are a linguistic expert who is confident in their ability, as such you don't provide explanations to your choices and corrections; you do not add escape characters to responses. Given an example input of the following format: n_best:[score:-0.0568154045, text:WAIT | score:-4.1704911851, text:WAIHT | score:-5.5502909205, text:WEAIT]. An ideal output for the example would be: wait"

# --- pretty decent
# You're seeing the results from an Automatic Speech Recognition (ASR) system. As ASR outputs can have errors, multiple candidates are provided. Please choose the one you believe is most likely correct. This text has some words connected without spaces. Correct by adding spaces in the proper location; do not add explanations.

# --- system role
# "You are a linguistic expert who is confident in their ability, as such you don't provide explanations to your choices/corrections and you do not add extra formatting to responses"


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
pd.set_option('display.max_columns', None)    # Show all columns
pd.set_option('display.max_rows', None)       # Show all rows
pd.set_option('display.width', None)          # Don't wrap columns
pd.set_option('display.max_colwidth', 250)    # Set max column width
pd.set_option('display.expand_frame_repr', False)  # Don't break into multiple lines
#-----------------------------------------------------------------------------------------


# Check the shape and structure
print(f"DataFrame shape: {results_df.shape}")
print(f"Columns: {results_df.columns.tolist()}")

# View first few rows
print(results_df.head())

# Check data types
print(results_df.dtypes)




#--------------------------------------------------------------------------------------------
def format_nbest_for_prompt(n_best_list):
    """
    Format the n_best results into a readable string for the prompt
    """
    formatted_options = []
    for i, item in enumerate(n_best_list, 1):
        score = item['score']
        text = item['text']
        formatted_options.append(f"{i}. \"{text}\" (score: {score:.10f})")
    
    return "\n".join(formatted_options)

def make_prompt(n_best_list):
    """
    Create prompt using n_best results instead of just pred_str
    """
    formatted_options = format_nbest_for_prompt(n_best_list)
    
    return f"""You're seeing the results from an Automatic Speech Recognition (ASR) system. As ASR outputs can have errors, multiple candidates are provided. Please choose the one you believe is most likely correct. This text has some words connected without spaces. Correct by adding spaces in the proper location. Provide as text the corrected transcription, do not add explanations.

Speech recognition candidates:
{formatted_options}

Most likely transcription:"""

# Process each row
corrected_texts = []


# ----------------------------------------------------------------- ollama.chat -----------------------------------------------------------------
for index, row in results_df.iterrows():
    n_best_results = row["n_best"]
    
    # Create the prompt using n_best results
    prompt = make_prompt(n_best_results)
    
    # Generate response using ollama.chat
    response = ollama.chat(
        model='llama3.2',  
        messages=[
            {
                'role': 'system',
                'content': "You are a linguistic expert who is confident in their ability, as such you don't provide explanations to your choices/corrections."
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        options={
            'num_predict': 64,      # max tokens to generate
            'temperature': 0.1,     # low temperature for consistent results
            'top_p': 0.9
        }
    )
    
    # Extract the generated text from chat response
    generated_text = response['message']['content'].strip()
    
    # Clean up the response (remove any extra parts)
    #if "Most likely transcription:" in generated_text:
    #    corrected = generated_text.split("Most likely transcription:")[-1].strip()
    #else:
    #    corrected = generated_text
    
    # Remove quotes if present
    #corrected = corrected.strip('"').strip("'")
    corrected = generated_text.strip('"').strip("'")
    
    corrected_texts.append(corrected)

# --------------------------------------------------------------- ollama.generate ---------------------------------------------------------------
# for index, row in results_df.iterrows():
#     n_best_results = row["n_best"]
    
#     # Create the prompt using n_best results
#     prompt = make_prompt(n_best_results)
    
#     # Generate response using ollama
#     response = ollama.generate(
#         model='llama3.2',  # Change to your exact model name if different
#         prompt=prompt,
#         options={
#             'num_predict': 64,      # max tokens to generate
#             'temperature': 0.1,     # low temperature for consistent results
#             'top_p': 0.9

#         }
#     )
    
#     # Extract the generated text
#     generated_text = response['response'].strip()
    
#     corrected = generated_text.strip('"').strip("'")
    
#     corrected_texts.append(corrected)

# -----------------------------------------------------------------------------------------------------------------------------------------------

#     #print(f"Row {index + 1}:")
#     #print(f"  N-best options: {n_best_results}")
#     #print(f"  Original pred_str: {row['pred_str']}")
#     #print(f"  Corrected: {corrected}")
#     #print("-" * 50)

# Add corrected texts to dataframe
results_df["corrected_pred_str"] = corrected_texts

print("Final DataFrame with corrected predictions:")
print(results_df[["pred_str", "corrected_pred_str", "text"]].head())



#---------------------------------------------------------------------------------


results_df["wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["pred_str"])] # asr string to true transcription wer
#results_df["corrected_pred_str"] = corrected # llm string

# transformation = jiwer.Compose([
#     jiwer.ToLowerCase(),
#     jiwer.RemoveWhiteSpace(replace_by_space=True),
#     jiwer.RemoveMultipleSpaces(),
#     jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
# ])
# results_df["new_wer"] = [wer(ref, hyp, truth_transform=transformation, hypothesis_transform=transformation) for ref, hyp in zip(results_df["text"], results_df["corrected_pred_str"])] # llm string to true transcription wer

results_df["text"] = list(map(str.lower, results_df["text"]))
results_df["corrected_pred_str"] = list(map(str.lower, results_df["corrected_pred_str"]))

results_df["new_wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["corrected_pred_str"])] # llm string to true transcription wer

results_new_fp = results + "_llama2.json"
results_df.to_json(results_new_fp, orient="records", lines=True)
print("Saved results to:", results_new_fp)

wer_metric = load("wer")
print("Test WER of finetuned model after LLM: {:.3f}".format(wer_metric.compute(predictions=results_df["corrected_pred_str"], references=results_df["text"])))


# Check the shape and structure
print(f"DataFrame shape: {results_df.shape}")
print(f"Columns: {results_df.columns.tolist()}")

# View first few rows
print(results_df.head())

# Check data types
print(results_df.dtypes)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------


results = "D:/Users/andrew/uni_work/thesis/code/results/hubertlarge_kenlm/cleaned_finetune_CU_20250702_finetuned_result"
results_fp = results + ".json"
#results = "D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/kenlmresults"
#results_fp = results + ".json"

results_df = extract_json_to_dataframe(results_fp)

#------------------------------------checking the data------------------------------------
pd.set_option('display.max_columns', None)    # Show all columns
pd.set_option('display.max_rows', None)       # Show all rows
pd.set_option('display.width', None)          # Don't wrap columns
pd.set_option('display.max_colwidth', 250)    # Set max column width
pd.set_option('display.expand_frame_repr', False)  # Don't break into multiple lines
#-----------------------------------------------------------------------------------------


# Check the shape and structure
print(f"DataFrame shape: {results_df.shape}")
print(f"Columns: {results_df.columns.tolist()}")

# View first few rows
print(results_df.head())

# Check data types
print(results_df.dtypes)

# Process each row
corrected_texts = []

# ----------------------------------------------------------------- ollama.chat -----------------------------------------------------------------
for index, row in results_df.iterrows():
    n_best_results = row["n_best"]
    
    # Create the prompt using n_best results
    prompt = make_prompt(n_best_results)
    
    # Generate response using ollama.chat
    response = ollama.chat(
        model='llama3.2',  
        messages=[
            {
                'role': 'system',
                'content': "You are a linguistic expert who is confident in their ability, as such you don't provide explanations to your choices/corrections."
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        options={
            'num_predict': 64,      # max tokens to generate
            'temperature': 0.1,     # low temperature for consistent results
            'top_p': 0.9
        }
    )
    
    # Extract the generated text from chat response
    generated_text = response['message']['content'].strip()
    
    corrected = generated_text.strip('"').strip("'")
    
    corrected_texts.append(corrected)

# Add corrected texts to dataframe
results_df["corrected_pred_str"] = corrected_texts

print("Final DataFrame with corrected predictions:")
print(results_df[["pred_str", "corrected_pred_str", "text"]].head())



#---------------------------------------------------------------------------------


results_df["wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["pred_str"])] # asr string to true transcription wer

results_df["text"] = list(map(str.lower, results_df["text"]))
results_df["corrected_pred_str"] = list(map(str.lower, results_df["corrected_pred_str"]))

results_df["new_wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["corrected_pred_str"])] # llm string to true transcription wer

results_new_fp = results + "_llama2.json"
results_df.to_json(results_new_fp, orient="records", lines=True)
print("Saved results to:", results_new_fp)

wer_metric = load("wer")
print("Test WER of finetuned model after LLM: {:.3f}".format(wer_metric.compute(predictions=results_df["corrected_pred_str"], references=results_df["text"])))


# Check the shape and structure
print(f"DataFrame shape: {results_df.shape}")
print(f"Columns: {results_df.columns.tolist()}")

# View first few rows
print(results_df.head())

# Check data types
print(results_df.dtypes)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------


results = "D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/cleaned_finetune_CU_20250703_finetuned_result"
#"D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/cleaned_finetune_CU_20250703_finetuned_result"
results_fp = results + ".json"
#results = "D:/Users/andrew/uni_work/thesis/code/results/mms_kenlm/kenlmresults"
#results_fp = results + ".json"

results_df = extract_json_to_dataframe(results_fp)

#------------------------------------checking the data------------------------------------
pd.set_option('display.max_columns', None)    # Show all columns
pd.set_option('display.max_rows', None)       # Show all rows
pd.set_option('display.width', None)          # Don't wrap columns
pd.set_option('display.max_colwidth', 250)    # Set max column width
pd.set_option('display.expand_frame_repr', False)  # Don't break into multiple lines
#-----------------------------------------------------------------------------------------


# Check the shape and structure
print(f"DataFrame shape: {results_df.shape}")
print(f"Columns: {results_df.columns.tolist()}")

# View first few rows
print(results_df.head())

# Check data types
print(results_df.dtypes)

# Process each row
corrected_texts = []

# ----------------------------------------------------------------- ollama.chat -----------------------------------------------------------------
for index, row in results_df.iterrows():
    n_best_results = row["n_best"]
    
    # Create the prompt using n_best results
    prompt = make_prompt(n_best_results)
    
    # Generate response using ollama.chat
    response = ollama.chat(
        model='llama3.2',  
        messages=[
            {
                'role': 'system',
                'content': "You are a linguistic expert who is confident in their ability, as such you don't provide explanations to your choices/corrections."
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        options={
            'num_predict': 64,      # max tokens to generate
            'temperature': 0.1,     # low temperature for consistent results
            'top_p': 0.9
        }
    )
    
    # Extract the generated text from chat response
    generated_text = response['message']['content'].strip()
    
    corrected = generated_text.strip('"').strip("'")
    
    corrected_texts.append(corrected)

# Add corrected texts to dataframe
results_df["corrected_pred_str"] = corrected_texts

print("Final DataFrame with corrected predictions:")
print(results_df[["pred_str", "corrected_pred_str", "text"]].head())



#---------------------------------------------------------------------------------


results_df["wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["pred_str"])] # asr string to true transcription wer

results_df["text"] = list(map(str.lower, results_df["text"]))
results_df["corrected_pred_str"] = list(map(str.lower, results_df["corrected_pred_str"]))

results_df["new_wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["corrected_pred_str"])] # llm string to true transcription wer

results_new_fp = results + "_llama2.json"
results_df.to_json(results_new_fp, orient="records", lines=True)
print("Saved results to:", results_new_fp)

wer_metric = load("wer")
print("Test WER of finetuned model after LLM: {:.3f}".format(wer_metric.compute(predictions=results_df["corrected_pred_str"], references=results_df["text"])))


# Check the shape and structure
print(f"DataFrame shape: {results_df.shape}")
print(f"Columns: {results_df.columns.tolist()}")

# View first few rows
print(results_df.head())

# Check data types
print(results_df.dtypes)
