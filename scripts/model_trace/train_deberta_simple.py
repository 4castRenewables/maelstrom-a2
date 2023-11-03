import logging
import os
import random

import a2.training.training_performance
import a2.utils
import numpy as np
import requests
import transformers
import xarray
from a2.training import benchmarks as timer

os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

# hyper parameters, with suggestions commented out
learning_rate = 3e-05  # 1e-5 8e-5 1e-4
batch_size = 32  # 16 64
weight_decay = 0.01  # 0.1 0.05
num_train_epochs = 1
warmup_ratio = 0
warmup_steps = 500  # 0 1000
hidden_dropout_prob = 0.1  # 0.2 0.3
cls_dropout = 0.1  # 0.2 0.3
lr_scheduler_type = "linear"  # "cosine"
fp16 = False  # True, only available when training on GPU

args = transformers.TrainingArguments(
    "output/",
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    warmup_steps=warmup_steps,
    lr_scheduler_type=lr_scheduler_type,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    fp16=fp16,
    disable_tqdm=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


def fake_dataset(n_sentences=20, max_words_per_sentence=20, min_words_per_sentence=3):
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"

    response = requests.get(word_site)
    WORDS = response.content.splitlines()
    WORDS = [w.decode("utf-8") for w in WORDS]

    text = []
    for _ in range(n_sentences):
        length_sentence = random.randrange(min_words_per_sentence, max_words_per_sentence)
        sentence = " ".join([random.choice(WORDS) for _ in range(length_sentence)])
        text.append(sentence)
    index = np.arange(n_sentences, dtype=np.int64)
    raining = np.array([random.choice([0, 1]) for i in range(n_sentences)], dtype=np.int64)
    return xarray.Dataset(
        data_vars=dict(
            text=(["index"], np.array(text, dtype=str)),
            raining=(["index"], raining),
        ),
        coords=dict(index=index),
    )


model_path = "microsoft/deberta-v3-base"  # , "microsoft/deberta-v3-small" for smaller version
ds_raw = fake_dataset()
dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(model_path)
dataset = dataset_object.build(ds_raw, np.arange(1, ds_raw.index.shape[0] - 1), [0])

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

tmr = timer.Timer()

trainer = a2.training.training_performance.TrainerWithTimer(
    model=model,
    args=args,
    callbacks=[
        a2.training.training_deep500.TimerCallback(tmr, gpu=True),
    ],
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
