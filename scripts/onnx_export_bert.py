import torch
import transformers


model_folder = "/home/kristian/Desktop/ap2_trained"
model_folder = "/home/kristian/Projects/a2/models/deberta-v3-small"
tweets_filename = "~/Desktop/2020_text_precipitation.nc"


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_folder,
    use_fast=False,
)

dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_folder, torchscript=True)

torch.onnx.export(
    model,
    tuple(dummy_model_input.values()),
    f="torch-model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    },
    do_constant_folding=True,
    opset_version=13,
)
