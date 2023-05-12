import torch
import transformers


model_folder = "/home/kristian/Desktop/ap2_trained"
model_folder = "/home/kristian/Projects/a2/models/deberta-v3-small"
tweets_filename = "~/Desktop/2020_text_precipitation.nc"

text = "It is raining"
indexed_tokens = [1, 325, 269, 19938, 2] + [0] * 164

input_ids = torch.tensor([indexed_tokens])
attention_mask = torch.zeros_like(input_ids)
token_type_ids = torch.zeros_like(input_ids)

dummy_input = [input_ids, attention_mask, token_type_ids]

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_folder, torchscript=True)

traced_model = torch.jit.trace(model, dummy_input)
torch.jit.save(traced_model, "/home/kristian/Desktop/traced_ap2.pt")
