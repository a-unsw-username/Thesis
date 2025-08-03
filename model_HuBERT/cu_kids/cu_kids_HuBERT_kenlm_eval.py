#!pip install datasets==1.18.3
#!pip install transformers==4.17.0
#!pip install jiwer


# ------------------------------------------
#       Import required packages
# ------------------------------------------
# For printing filepath
import os
# ------------------------------------------
print('Running: ', os.path.abspath(__file__))
# ------------------------------------------
# For accessing date and time
from datetime import date
from datetime import datetime
now = datetime.now()
# Print out dd/mm/YY H:M:S
# ------------------------------------------
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Started:", dt_string)
# ------------------------------------------ 
print("\n------> IMPORTING PACKAGES.... ---------------------------------------\n")
print("-->Importing datasets...")
# Import datasets and evaluation metric
from datasets import load_dataset, ClassLabel
from datasets import load_from_disk
#from datasets import set_caching_enabled, set_caching_directory
print("-->Importing evaluate...")
from evaluate import load
# Convert pandas dataframe to DatasetDict
from datasets import Dataset
# WER metric for per row output evaluation
from jiwer import wer
print("-->Importing KenLM dependencies...")
from pyctcdecode import BeamSearchDecoderCTC
from pyctcdecode import build_ctcdecoder
import kenlm
# Generate random numbers
print("-->Importing random...")
import random
from IPython.display import display, HTML
import string
# Manipulate dataframes and numbers
print("-->Importing pandas & numpy...")
import pandas as pd
import numpy as np
# Use regex
print("-->Importing re...")
import re
# Read, Write, Open json files
print("-->Importing json...")
import json
# Use models and tokenizers
print("-->Importing Wav2VecCTC...")
from transformers import Wav2Vec2CTCTokenizer
from transformers import HubertForCTC
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
# Loading audio files
print("-->Importing soundfile...")
import soundfile as sf
print("-->Importing librosa...")
import librosa
# For training
print("-->Importing torch, dataclasses & typing...")
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
print("-->Importing from transformers for training...")
from transformers import TrainingArguments
from transformers import Trainer
print("-->Importing pyarrow for loading dataset...")
import pyarrow as pa
import pyarrow.csv as csv
print("-->SUCCESS! All packages imported.")

# ------------------------------------------
#      Setting experiment arguments
# ------------------------------------------
print("\n------> EXPERIMENT ARGUMENTS ----------------------------------------- \n")


# MMS next
# then waveLM


# Perform Training (True/False)
# If false, this will go straight to model evaluation
training = False
print("training:", training)

# Continue from checkpoint?
use_checkpoint = True
print("Using checkpoint:", use_checkpoint)

# Evaluate the base model (without own training)?
eval_base = False
print("Evaluating base model:", eval_base)

# Use a pretrained tokenizer (True/False)
#     True: Use existing tokenizer (if custom dataset has same vocab)
#     False: Use custom tokenizer (if custom dataset has different vocab)
use_pretrained_tokenizer = True
print("use_pretrained_tokenizer:", use_pretrained_tokenizer)

# Set tokenizer
pretrained_tokenizer = "facebook/wav2vec2-large-960h"
if use_pretrained_tokenizer:
    print("pretrained_tokenizer:", pretrained_tokenizer)

# save results and model with identifying name
base_fp = "/srv/scratch/z5363024/thesis/"
model = "HuBERT_Large_kenlm"
dataset_name = "CU"

# base/finetune _ dataset _ YearMonthDate
testing_id = "finetune_CU_20250710"

model_dir = base_fp + model + '/model/' + dataset_name + '/'
os.makedirs(model_dir, exist_ok=True)
model_fp =  model_dir + testing_id
print("--> model_fp:", model_fp)

final_fp = model_fp + '/final'
print("--> final_fp:", final_fp)

vocab_dir = base_fp + model + '/vocab/' + dataset_name + '/'
os.makedirs(vocab_dir, exist_ok=True)
vocab_fp =  vocab_dir + testing_id + '_vocab.json'
print("--> vocab_fp:", vocab_fp)

# Path to save results output
#if eval_base:
#    results_fp = base_fp + model + '/baseline_result/' + dataset_name + '/' + testing_id + '_results.csv'
#    print("--> results_fp:", results_fp) 
#else:
#    results_fp = base_fp + model + '/finetuned_result/' + dataset_name + '/' + testing_id + '_results.csv'
#    print("--> results_fp:", results_fp)

baseline_results_dir = base_fp + model + '/baseline_result/' + dataset_name + '/'
os.makedirs(baseline_results_dir, exist_ok=True)
baseline_results_fp = baseline_results_dir + testing_id + '_baseline_result.json'
print("--> baseline_results_fp:", baseline_results_fp)

finetuned_results_dir = base_fp + model + '/finetuned_result/' + dataset_name + '/'
os.makedirs(finetuned_results_dir, exist_ok=True)
finetuned_results_fp = finetuned_results_dir + testing_id + '_finetuned_result.json'
print("--> finetuned_results_fp:", finetuned_results_fp)

#set_caching_enabled = True
#set_caching_directory = "/srv/scratch/speechdata/andrew_c/cache_dir"


#data_cache_fp = '/srv/scratch/z5363024/.cache/huggingface/datasets/timit'
data_dict_fp = '/srv/scratch/speechdata/speech-corpora/children/TD/cu_dataset'
data_dict_save_fp = '/srv/scratch/speechdata/andrew_c'

checkpoint = "/srv/scratch/z5363024/thesis/HuBERT_Large_kenlm/model/CU/finetune_CU_20250702/final"
base_model = "facebook/hubert-large-ll60k"
base_model_pretrained = "facebook/hubert-large-ls960-ft"




print("\n------> MODEL ARGUMENTS... -------------------------------------------\n")
# For setting model = HuBERTForCTC.from_pretrained()

set_hidden_dropout = 0.1                    # Default = 0.1
print("hidden_dropout:", set_hidden_dropout)
set_activation_dropout = 0.1                # Default = 0.1
print("activation_dropout:", set_activation_dropout)
set_attention_dropout = 0.1                 # Default = 0.1
print("attention_dropoutput:", set_attention_dropout)
set_feat_proj_dropout = 0.1                 # Default = 0.1
print("feat_proj_dropout:", set_feat_proj_dropout)
set_layerdrop = 0.1                         # Default = 0.1
print("layerdrop:", set_layerdrop)
set_mask_time_prob = 0.05                  # Default = 0.05
print("mask_time_prob:", set_mask_time_prob)
set_mask_time_length = 10                   # Default = 10
print("mask_time_length:", set_mask_time_length)
set_ctc_loss_reduction = "mean"             # Default = "sum"
print("ctc_loss_reduction:", set_ctc_loss_reduction)
set_ctc_zero_infinity = True               # Default = False
print("ctc_zero_infinity:", set_ctc_zero_infinity)
set_gradient_checkpointing = True           # Default = False
print("gradient_checkpointing:", set_gradient_checkpointing)



print("\n------> TRAINING ARGUMENTS... ----------------------------------------\n")
# For setting training_args = TrainingArguments()

set_evaluation_strategy = "no"           # Default = "no"
print("evaluation strategy:", set_evaluation_strategy)
set_per_device_train_batch_size = 8         # Default = 8
print("per_device_train_batch_size:", set_per_device_train_batch_size)
set_gradient_accumulation_steps = 1         # Default = 1
print("gradient_accumulation_steps:", set_gradient_accumulation_steps)
set_learning_rate = 0.00005                 # Default = 0.00005
# change this around
# 
print("learning_rate:", set_learning_rate)
set_weight_decay = 0.005                     # Default = 0 | prev 0.01
print("weight_decay:", set_weight_decay)
set_adam_beta1 = 0.9                        # Default = 0.9
print("adam_beta1:", set_adam_beta1)
set_adam_beta2 = 0.98                       # Default = 0.999
print("adam_beta2:", set_adam_beta2)
set_adam_epsilon = 0.00000001               # Default = 0.00000001
print("adam_epsilon:", set_adam_epsilon)
set_num_train_epochs = 10                   # Default = 3.0
print("num_train_epochs:", set_num_train_epochs)
set_max_steps = -1                          # Default = -1, overrides epochs
print("max_steps:", set_max_steps)
set_lr_scheduler_type = "linear"            # Default = "linear"
print("lr_scheduler_type:", set_lr_scheduler_type )
set_warmup_ratio = 0.15                      # Default = 0.0
# between 0.1 to 0.2
# 0.005 increments
print("warmup_ratio:", set_warmup_ratio)
set_logging_strategy = "steps"              # Default = "steps"
print("logging_strategy:", set_logging_strategy)
set_logging_steps = 1000                      # Default = 500
print("logging_steps:", set_logging_steps)
set_save_strategy = "steps"                 # Default = "steps"
print("save_strategy:", set_save_strategy)
set_save_steps = 1000                         # Default = 500
print("save_steps:", set_save_steps)
set_save_total_limit = 2                   # Optional                 
print("save_total_limit:", set_save_total_limit)
set_fp16 = True                             # Default = False
print("fp16:", set_fp16)
set_eval_steps = 1000                         # Optional
print("eval_steps:", set_eval_steps)
set_load_best_model_at_end = False           # Default = False
print("load_best_model_at_end:", set_load_best_model_at_end)
set_metric_for_best_model = "wer"           # Optional
print("metric_for_best_model:", set_metric_for_best_model)
set_greater_is_better = False               # Optional
print("greater_is_better:", set_greater_is_better)
set_group_by_length = True                  # Default = False
print("group_by_length:", set_group_by_length)



#timit = load_dataset("timit_asr", cache_dir=data_cache_fp)
#timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

#data = load_from_disk(data_dict_fp)
#data.save_to_disk(data_dict_save_fp)
data = load_from_disk(data_dict_save_fp)
#data.cleanup_cache_files() #creates cache after every operation to data
data = data.remove_columns(["age", "speaker_id"])

# display random samples
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

show_random_elements(data["train"].remove_columns(["audio"]), num_examples=10)

# remove special characters and make uniform case

def process_transcription(batch):
    #batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper()
    #batch["text"] = batch["text"].replace("<unk>", "[UNK]")
    batch["text"] = batch["text"].upper()
    batch["text"] = batch["text"].replace("<UNK>", "<unk>")
    return batch

data = data.map(process_transcription)

#def extract_all_chars(batch):
#    all_text = " ".join(batch["text"])
#    vocab = list(set(all_text))
#    return {"vocab": [vocab], "all_text": [all_text]}

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    
    # Keep only uppercase letters, space, apostrophe
    allowed_chars = set(string.ascii_uppercase + " '")  
    
    # Filter out unwanted characters
    vocab = sorted(set(c for c in all_text if c in allowed_chars))
    
    return {"vocab": [vocab], "all_text": [all_text]}
    
if not use_pretrained_tokenizer:
    print("--> Creating map(...) function for vocab...")
    vocabs = data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=data.column_names["train"])
    # Create union of all distinct letters in train and test set
    # and convert resulting list into enumerated dictionary
    # Vocab includes a-z, ' , space, UNK, PAD
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    print("--> Vocab len:", len(vocab_dict), "\n", vocab_dict)
    # Give space " " a visible character " | "
    # Include "unknown" [UNK] token for dealing with characters
    # not encountered in training.
    # Add padding token to corresponds to CTC's "blank token".
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["<unk>"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print("--> Vocab len:", len(vocab_dict), "\n", vocab_dict)
    # Save vocab as a json file
    with open(vocab_fp, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    print("SUCCESS: Created vocabulary file at", vocab_fp)
    # Use json file to instantiate an object of the 
    # Wav2VecCTCTokenziser class

if use_pretrained_tokenizer:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_tokenizer)
else:
    tokenizer = Wav2Vec2CTCTokenizer(vocab_fp, unk_token="<unk>", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# checking audio
#import IPython.display as ipd
#import numpy as np
#import random

#rand_int = random.randint(0, len(data["train"]))

#print(data["train"][rand_int]["text"])
#ipd.Audio(data=np.asarray(data["train"][rand_int]["audio"]["array"]), autoplay=True, rate=16000)

# checking sample
rand_int = random.randint(0, len(data["train"]))

print("Target text:", data["train"][rand_int]["text"])
print("Input array shape:", np.asarray(data["train"][rand_int]["audio"]["array"]).shape)
print("Sampling rate:", data["train"][rand_int]["audio"]["sampling_rate"])

def prepare_dataset(batch):
    audio = batch["audio"]

    # Ensure transcription is not empty before processing
    #if not batch["text"].strip():
    #    return None  # This removes the sample if text is empty

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


# filter out empty text
#for split in data.keys():
#    if "text" in data[split].column_names:
#        data[split] = data[split].filter(lambda x: x["text"] is not None and x["text"].strip() != "")
data = data.filter(lambda x: x["text"].strip() != "")


data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=4)


# everything above max_input_length seconds is filtered out to minimise memory
max_input_length_in_sec = 10.0
data["train"] = data["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])


# padding to length of max input
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


#-----------------------------------------------------------------------------------------------
#KenLM decoder

full_vocab = processor.tokenizer.get_vocab()

#----error checking-----
#print("Sample vocab_dict entries:")
#for k, v in list(vocab_dict.items())[:10]:
#    print(f"{k}: {v} (type: {type(v)})")
#-----------------------
vocab_dict = full_vocab

#sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])
sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])

vocab_list = [token.replace("|", " ") if token != "[PAD]" else "" for token, _ in sorted_vocab]

kenlm_path = "/srv/scratch/z5363024/thesis/kenlm_old/build/cu_kids_lm.arpa"

decoder = build_ctcdecoder(
    vocab_list,
    kenlm_model_path=kenlm_path,
    alpha=0.5,  # LM weight
    beta=1.0    # Word insertion penalty
)

print("Final vocab list for KenLM decoding:", vocab_list)



#-----------------------------------------------------------------------------------------------


wer_metric = load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



#if use_checkpoint == True:
    #pretrained_model = checkpoint
if eval_base == True:
    pretrained_model = base_model_pretrained
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(base_model_pretrained)
elif use_checkpoint == True:
    pretrained_model = checkpoint
else:
    pretrained_model = base_model


# Load model
model = HubertForCTC.from_pretrained(
    pretrained_model,
    vocab_size=len(processor.tokenizer),
    hidden_dropout=set_hidden_dropout,
    activation_dropout=set_activation_dropout,
    attention_dropout=set_attention_dropout,
    feat_proj_dropout=set_feat_proj_dropout,
    layerdrop=set_layerdrop,
    mask_time_prob=set_mask_time_prob,
    mask_time_length=set_mask_time_length,
    ctc_loss_reduction=set_ctc_loss_reduction,
    ctc_zero_infinity=set_ctc_zero_infinity,
    #gradient_checkpointing=set_gradient_checkpointing,
    pad_token_id=processor.tokenizer.pad_token_id
)

model.freeze_feature_encoder()


training_args = TrainingArguments(
  output_dir=model_fp,
  evaluation_strategy=set_evaluation_strategy,
  per_device_train_batch_size=set_per_device_train_batch_size,
  gradient_accumulation_steps=set_gradient_accumulation_steps,
  learning_rate=set_learning_rate,
  weight_decay=set_weight_decay,
  adam_beta1=set_adam_beta1,
  adam_beta2=set_adam_beta2,
  adam_epsilon=set_adam_epsilon,
  num_train_epochs=set_num_train_epochs,
  max_steps=set_max_steps,
  lr_scheduler_type=set_lr_scheduler_type,
  warmup_ratio=set_warmup_ratio,
  logging_strategy=set_logging_strategy,
  logging_steps=set_logging_steps,
  save_strategy=set_save_strategy,
  save_steps=set_save_steps,
  save_total_limit=set_save_total_limit,
  fp16=set_fp16,
  eval_steps=set_eval_steps,
  load_best_model_at_end=set_load_best_model_at_end,
  metric_for_best_model=set_metric_for_best_model,
  greater_is_better=set_greater_is_better,
  group_by_length=set_group_by_length
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=processor.feature_extractor,
)

# --------------- continue from checkpoint errors --------------------

#torch.serialization.add_safe_globals([torch._utils._rebuild_tensor_v2])

#from torch.serialization import safe_globals
#safe_globals(["numpy.core.multiarray._reconstruct"])

# Manually load RNG state with weights_only=False
#def load_rng_state_fix(rng_file):
#    return torch.load(rng_file, weights_only=False)  # Forces full checkpoint loading

# Override Trainer's _load_rng_state method
#trainer._load_rng_state = load_rng_state_fix
if use_checkpoint:
    import torch
    from torch.serialization import add_safe_globals
    import numpy.core

    # Allow numpy reconstruct (required for legacy RNG states)
    add_safe_globals([numpy.core.multiarray._reconstruct])

    # Force loading with weights_only=False
    def load_rng_state_fix(checkpoint_dir):
        rng_file = os.path.join(checkpoint_dir, "rng_state.pth")
        if not os.path.exists(rng_file):
            raise FileNotFoundError(f"Expected rng_state.pth in checkpoint directory: {checkpoint_dir}")
        return torch.load(rng_file, weights_only=False)

    trainer._load_rng_state = load_rng_state_fix


# --------------------------------------------------------------------


# ------------------------------------------
#               Training
# ------------------------------------------

# can only trainer.train() if using own new model or own model checkpoint

if training:
    print("\n------> STARTING TRAINING... ----------------------------------------- \n")
    torch.cuda.empty_cache()
    # Train
    if use_checkpoint:
        #trainer.train(resume_from_checkpoint=pretrained_model)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # Save the model
    model.save_pretrained(final_fp)
    processor.save_pretrained(final_fp)
elif eval_base:
    print("\n------> SKIPPING TRAINING ON BASE MODEL... ----------------------------------------- \n")
else: 
    print("\n------> SKIPPING TRAINING OF PRETRAINED MODEL... ----------------------------------------- \n")



# ------------------------------------------
#            Evaluation
# ------------------------------------------
# 

# eval_base for not trained, else eval own trained model

print("\n------> EVALUATING MODEL... ------------------------------------------ \n")
torch.cuda.empty_cache()

#---------------------------------------Regular map_to_result----------------------------------------

# function to return pred_str and text in place of 'test'
#def map_to_result(batch):
#    with torch.no_grad():
#        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
#        logits = model(input_values).logits

#    pred_ids = torch.argmax(logits, dim=-1)
#    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
#    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

#    return batch
#----------------------------------------------------------------------------------------------------

#---------------------------------------KenLM map_to_result------------------------------------------
def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits.cpu().numpy()[0]

    beam_results = decoder.decode(logits)
    #beam_results = decoder.decode_beams(logits, beam_width=5)

    n_best_hypotheses = []
    
    # for i, beam in enumerate(beam_results[:5]):
    #     if isinstance(beam, tuple):
    #         text = str(beam[0])
    #         score = 0.0
    #         for item in beam[1:]:
    #             if isinstance(item, (float, int)):
    #                 score = float(item)
    #                 break
    #         n_best_hypotheses.append({"text": text, "score": score})
    #     elif hasattr(beam, "text") and hasattr(beam, "lm_score"):
    #         n_best_hypotheses.append({
    #             "text": str(beam.text), 
    #             "score": float(beam.lm_score)
    #         })
    #     else:
    #         n_best_hypotheses.append({"text": str(beam), "score": 0.0})

    # Store as structured list of dictionaries
    batch["pred_str"] = beam_results
    # batch["n_best"] = n_best_hypotheses
    # batch["pred_str"] = n_best_hypotheses[0]["text"] if n_best_hypotheses else ""
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
    return batch

#----------------------------------------------------------------------------------------------------

# if evaluating trained model
if training:
    processor = Wav2Vec2Processor.from_pretrained(final_fp)
    model = HubertForCTC.from_pretrained(final_fp).to("cuda")
    
    model.eval()

    results = data["test"].map(map_to_result, remove_columns=data["test"].column_names)
    # Ensure there are no empty references before calculating WER
    filtered_results = results.filter(lambda x: x["text"].strip() != "")
    results_df = filtered_results.to_pandas()
    #results_df.to_json(finetuned_results_fp, orient="records", lines=True)

    #should be of the format: 
    #{
    #  "text": "ground truth",
    #  "pred_str": "best decoded output",
    #  "n_best": [
    #    "best decoded output",
    #    "next best hypothesis",
    #    "third best hypothesis"
    #  ]
    #}

    results_df["wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["pred_str"])]
    results_df.to_json(finetuned_results_fp, orient="records", lines=True)
    print("Saved results to:", finetuned_results_fp)

    #show_random_elements(results) # checking

    print("Test WER of trained model: {:.3f}".format(wer_metric.compute(predictions=filtered_results["pred_str"], references=filtered_results["text"])))




# if evaluating base model
if eval_base:
    torch.cuda.empty_cache()
    processor = Wav2Vec2Processor.from_pretrained(base_model_pretrained)
    model = HubertForCTC.from_pretrained(base_model_pretrained).to("cuda")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(base_model_pretrained)
    model.eval()

    results = data["test"].map(map_to_result, remove_columns=data["test"].column_names)
    # Ensure there are no empty references before calculating WER
    filtered_results = results.filter(lambda x: x["text"].strip() != "")
    results_df = filtered_results.to_pandas()

    results_df["wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["pred_str"])]

    results_df.to_json(baseline_results_fp, orient="records", lines=True)
    print("Saved results to:", baseline_results_fp)

    show_random_elements(results) # checking

    print("Test WER of baseline model: {:.3f}".format(wer_metric.compute(predictions=filtered_results["pred_str"], references=filtered_results["text"])))




# eval model without training
else:
    torch.cuda.empty_cache()
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = HubertForCTC.from_pretrained(checkpoint).to("cuda")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(checkpoint)
    model.eval()

    results = data["test"].map(map_to_result, remove_columns=data["test"].column_names)
    # Ensure there are no empty references before calculating WER
    filtered_results = results.filter(lambda x: x["text"].strip() != "")
    results_df = filtered_results.to_pandas()

    results_df["wer"] = [wer(ref, hyp) for ref, hyp in zip(results_df["text"], results_df["pred_str"])]

    results_df.to_json(finetuned_results_fp, orient="records", lines=True)
    print("Saved results to:", finetuned_results_fp)

    show_random_elements(results) # checking

    print("Test WER of finetuned model: {:.3f}".format(wer_metric.compute(predictions=filtered_results["pred_str"], references=filtered_results["text"])))
