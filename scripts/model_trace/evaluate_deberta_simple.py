import datasets
import numpy as np
import torch
import transformers
import xarray


def _tok_func(tokenizer, x: dict):
    return tokenizer(x["inputs"], padding=True)


def get_trainer(tokenizer, model_folder):
    args = transformers.TrainingArguments(
        "./",
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_folder)
    trainer = transformers.Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
    )
    return trainer


def tokenize_dataset(ds, model_folder, key_inputs, key_label):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_folder,
        use_fast=False,
    )

    df = ds[[key_inputs, key_label]].to_pandas()
    columns = {key_inputs: "inputs", key_label: "label"}
    df = df.rename(columns=columns)
    datasets_ds = datasets.Dataset.from_pandas(df)
    tok_ds = datasets_ds.map(lambda x: _tok_func(tokenizer, x), batched=True)
    return tok_ds, tokenizer


model_folder = "/home/kristian/Desktop/ap2_trained"
tweets_filename = "~/Desktop/2020_text_precipitation.nc"
key_inputs = "text_normalized"
key_label = "raining"

ds = xarray.load_dataset(tweets_filename).sel(index=range(100))
ds["raining"] = (["index"], np.array(ds["tp_h"].values > 1e-8, dtype=int))

tok_ds, tokenizer = tokenize_dataset(ds, model_folder, key_inputs, key_label)

trainer = get_trainer(tokenizer, model_folder)
prediction_probabilities = torch.nn.functional.softmax(torch.Tensor(trainer.predict(tok_ds).predictions), dim=1).numpy()
predictions = prediction_probabilities.argmax(-1)
