from transformers import AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

llm_model_name = "meta-llama/Meta-Llama-3.2-3B-Instruct"


# ------------------------------------------
#            LLM Adjusted Evaluation
# ------------------------------------------
# 

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# Load model with 4-bit quantization (efficient for 1 GPU)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

# Create generation pipeline
llm = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, device_map="auto")


def make_prompt(text):
    return f"""You are an expert in transcribing children's speech. Fix any spelling or grammar errors in the following sentence and return only the corrected version. If you do not correct a given piece of speech, simply return the original text. Correct only. Do not include explanations or additional comments.

Original: "{text}"
Corrected:"""

# Generate corrections
corrected = []
for original in results_df["pred_str"]:
    prompt = make_prompt(original)
    output = llm(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
    # Extract only the corrected part (after "Corrected:")
    correction = output.split("Corrected:")[-1].strip()
    corrected.append(correction)

#results_df["corrected_pred_str"] = corrected # llm string
#results_df["new_wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["corrected_pred_str"])] # llm string to true transcription wer
#results_df["wer_improvement"] = results_df["wer"] - results_df["new_wer"]

#if training:
#    results_df.to_csv(finetuned_adjusted_results_fp)
#    print("Saved results to:", finetuned_adjusted_results_fp)
#else:
#    results_df.to_csv(baseline_adjusted_results_fp)
#    print("Saved results to:", baseline_adjusted_results_fp)