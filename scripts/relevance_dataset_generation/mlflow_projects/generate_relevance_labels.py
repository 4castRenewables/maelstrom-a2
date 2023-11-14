import logging

import a2.dataset.load_dataset
import a2.dataset.utils_dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.training_hugging
import a2.utils.argparse
import numpy as np
import transformers
import torch
import re

import a2.utils

import a2.training.training_hugging
import a2.training.evaluate_hugging
import a2.training.dataset_hugging
import a2.plotting.analysis
import a2.plotting.histograms
import a2.dataset

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def main(args):
    """Split labeled Tweets into train, test, validation set"""
    logging.info(f"Args used: {args.__dict__}")

    ds_raw = a2.dataset.load_dataset.load_tweets_dataset(args.filename_tweets, raw=True)

    logging.info(f"Dataset contains {ds_raw.index.shape[0]=} Tweets.")
    logging.info(f"Available GPUs:\n{[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")

    ds = a2.dataset.load_dataset.reset_index_coordinate(ds_raw)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.folder_model, device_map="auto", trust_remote_code=True, quantization_config=bnb_config
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.folder_model)
    tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"{transformers.__version__=}")

    prompt = r"""
    Read below Tweets and tell me if they say that it is raining or sunny.
    Format your answer in a human readable way,

    Tweets:
    Tweet 1: "The sound of rain tapping on the window"
    Tweet 2: "Boris likes drinking water".
    """

    example_output = """
    Return the results in a json file like: [
    { "tweet": 1, "content": "The sound of rain tapping on the window", "explanation": "The sound of rain heard implies that is raining.", "score": 0.9 },
    { "tweet": 2, "content": "Boris likes drinking water", "explanation": "The Tweet does not mention any information related to presence of rain or sun.", "score": 0.1},
    { "tweet": 3, "content": ...
    ]

    Result: [ { "tweet": 1, "content":"""
    ds_no_snow = ds.where(~ds.text_normalized.str.contains("snow", flags=re.IGNORECASE), drop=True)

    n_start = int(args.n_start)
    n_sample = int(args.n_samples)

    logging.info(f"Selecting {n_sample=} Tweets starting with index {n_start=}.")

    tweets = ds_no_snow["text_normalized"].values[slice(n_start, n_start + n_sample)]

    logging.info(f"Selected {len(tweets)} Tweets for relevance label generation.")

    for tweet_sample in np.array_split(tweets, len(tweets) // 5):
        prediction = generate_prediction(tokenizer, model, prompt, tweet_sample, example_output)
        with open("dump_relevance.csv", "a") as fd:
            fd.write(prediction)


def string_tweets(tweets):
    string = ""
    for i_tweet, t in enumerate(tweets):
        string += f'Tweet {i_tweet + 3}: "{t}"\n'
    return string


def tokenize_prompt(tokenizer, prompt):
    return tokenizer.encode(prompt, return_tensors="pt").cuda()


def generate_prediction(tokenizer, model, prompt, tweets, example_output):
    full_prompt = prompt + string_tweets(tweets) + example_output
    input_ids = tokenize_prompt(tokenizer, full_prompt)

    sample_outputs = model.generate(
        input_ids,
        temperature=0.9,
        # do_sample=True,
        max_length=650,
        top_k=50,
        # top_p=0.95,
        # num_return_sequences=3
    )

    for i, sample_output in enumerate(sample_outputs):
        prediction = tokenizer.decode(sample_output, skip_special_tokens=True)
        print(f"{prompt=}")
        print("---------")
        print(f"prediction\n{prediction}")
    return prediction


def save_subsection_dataset(ds, filename, indices):
    ds_subsection = ds.sel(index=indices)
    a2.dataset.load_dataset.save_dataset(ds_subsection, filename=filename, reset_index=True, no_conversion=False)


if __name__ == "__main__":
    parser = a2.utils.argparse.get_parser()
    parser.add_argument(
        "--filename_tweets",
        type=str,
        default="/p/project/training2330/a2/data/bootcamp2023/tweets/tweets_2017_01_era5_normed_filtered.nc",
        help="Filename of training data.",
    )
    parser.add_argument(
        "--folder_model",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/models/falcon-40b",
        help="Path to model.",
    )
    parser.add_argument(
        "--n_start",
        type=int,
        default=0,
        help="Start inference from Tweet with thins index.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of tweets evaluated.",
    )
    args = parser.parse_args()
    main(args)
