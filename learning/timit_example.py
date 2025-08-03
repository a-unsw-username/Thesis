#!pip install datasets==1.18.3
#!pip install transformers==4.17.0
#!pip install jiwer

from datasets import load_dataset, load_metric
from datasets import ClassLabel
import random
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Trainer

model_fp = "/srv/scratch/z5363024/thesis/timit_test"
#model_fp = "D:/Users/andrew/uni_work/thesis/code/timit_test"
finetuned_results_fp = model_fp + 'finetuned_results.csv'
data_cache_fp = '/srv/scratch/z5363024/.cache/huggingface/datasets/timit'

timit = load_dataset("timit_asr", cache_dir=data_cache_fp)
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

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

show_random_elements(timit["train"].remove_columns(["audio", "file"]), num_examples=10)

# remove special characters and make lowercase
import re

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch

timit = timit.map(remove_special_characters)
show_random_elements(timit["train"].remove_columns(["audio", "file"])) # check text


def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

# not really sure if this stuff is necessary or was shown for learning
vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

# not sure if this stuff will work
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# checking audio
import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(timit["train"]))

print(timit["train"][rand_int]["text"])
ipd.Audio(data=np.asarray(timit["train"][rand_int]["audio"]["array"]), autoplay=True, rate=16000)

# checking sample
rand_int = random.randint(0, len(timit["train"]))

print("Target text:", timit["train"][rand_int]["text"])
print("Input array shape:", np.asarray(timit["train"][rand_int]["audio"]["array"]).shape)
print("Sampling rate:", timit["train"][rand_int]["audio"]["sampling_rate"])

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"])#, num_proc=4)

# everything above 4 seconds is filtered out to minimise memory
max_input_length_in_sec = 4.0
timit["train"] = timit["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])



# the tutorial site didn't explain this but I think its just a really confusing way to pad to length of max input
# because I don't know what is going on 
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

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)
model.freeze_feature_encoder()


# is said to be good for timit
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=model_fp,
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit["train"],
    eval_dataset=timit["test"],
    tokenizer=processor.feature_extractor,
)

### training

trainer.train()
model.save_pretrained(model_fp)
processor.save_pretrained(model_fp)
processor = Wav2Vec2Processor.from_pretrained(model_fp)
model = Wav2Vec2ForCTC.from_pretrained(model_fp).cuda()

model.eval()

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
  return batch

results = timit["test"].map(map_to_result, remove_columns=timit["test"].column_names)
print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

show_random_elements(results) # checking

#model.to("cuda")

#with torch.no_grad():
#  logits = model(torch.tensor(timit["test"][:1]["input_values"], device="cuda")).logits

#pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
#" ".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist()))''' # for understanding

### joyuenn's/renee's

#def map_to_result(batch):
#  model.to("cuda")
#  input_values = processor(
#      batch["speech"], 
#      sampling_rate=batch["sampling_rate"], 
#      return_tensors="pt"
#  ).input_values.to("cuda")
#
#  with torch.no_grad():
#    logits = model(input_values).logits
#
#  pred_ids = torch.argmax(logits, dim=-1)
#  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
#  
#  return batch

#results = timit["test"].map(map_to_result)
# Save results to csv
#results_df = results.to_pandas()
#results_df = results_df.drop(columns=['speech', 'sampling_rate'])
#results_df.to_csv(finetuned_results_fp)
#print("Saved results to:", finetuned_results_fp)

# Getting the WER
#print("--> Getting fine-tuned test results...")
#print("Fine-tuned Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], 
#      references=results["target_text"])))
#cer_metric = load_metric("cer")
#print("Fine-tuned Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], 
#      references=results["target_text"])))
#print('\n')
# Deeper look into model: running the first test sample through the model, 
# take the predicted ids and convert them to their corresponding tokens.
#print("--> Taking a deeper look...")
#model.to("cuda")
#input_values = processor(timit["test"][0]["speech"], sampling_rate=timit["test"][0]["sampling_rate"], return_tensors="pt").input_values.to("cuda")

#with torch.no_grad():
#  logits = model(input_values).logits

#pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
#print(" ".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())))