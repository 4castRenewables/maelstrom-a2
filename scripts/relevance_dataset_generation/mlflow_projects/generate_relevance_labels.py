import logging
import os

import a2.dataset.load_dataset
import a2.dataset.utils_dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.training_hugging
import a2.utils.argparse
import utils_scripts
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
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    """Split labeled Tweets into train, test, validation set"""
    logging.info(f"Args used: {args.__dict__}")
    path_output = utils_scripts._determine_path_output(args)
    path_figures = os.path.join(path_output, args.figure_folder)
    _, ds_raw = utils_scripts.get_dataset(
        args, args.filename_tweets, dataset_object=None, set_labels=True, build_hugging_dataset=False
    )

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
    tweets = ds_no_snow["text_normalized"].values

    for tweet_sample in np.array_split(tweets, len(tweets) // 5):
        print(f"{tweet_sample}")
        prediction = generate_prediction(args, tokenizer, model, prompt, tweets, example_output)
        with open("dump_relevance.csv", "a") as fd:
            fd.write(prediction)

    _plot(args, ds, path_figures, tracker)


def string_tweets(tweets):
    string = ""
    for i_tweet, t in enumerate(tweets):
        string += f'Tweet {i_tweet + 3}: "{t}"\n'
    return string


def tokenize_prompt(tokenizer, prompt):
    return tokenizer.encode(prompt, return_tensors="pt").cuda()


def generate_prediction(args, tokenizer, model, prompt, tweets, example_output):
    full_prompt = prompt + string_tweets(tweets) + example_output
    input_ids = tokenize_prompt(tokenizer, full_prompt)

    sample_outputs = model.generate(
        input_ids,
        temperature=0.9,
        # do_sample=True,
        max_length=650,
        top_k=50,
        top_p=0.95,
        # num_return_sequences=3
    )

    for i, sample_output in enumerate(sample_outputs):
        prediction = tokenizer.decode(sample_output, skip_special_tokens=True)
        print(f"{prompt=}")
        print("---------")
        print(f"prediction\n{prediction}")
    return prediction


def _plot(args, ds, path_figures, tracker):
    utils_scripts.plot_and_log_histogram(
        ds, key="raining", path_figures=path_figures, tracker=tracker, filename="raining_histogram.pdf"
    )
    utils_scripts.plot_and_log_histogram(
        ds, key=args.key_rain, path_figures=path_figures, tracker=tracker, filename="precipitation_histogram.pdf"
    )


def save_subsection_dataset(ds, filename, indices):
    ds_subsection = ds.sel(index=indices)
    a2.dataset.load_dataset.save_dataset(ds_subsection, filename=filename, reset_index=True, no_conversion=False)


def select_dataset(args, ds):
    if args.select_relevant:
        logging.info(f"Selecting all Tweets where {args.key_relevance}==True")
        logging.info(f"Before selection {ds.index.shape[0]} Tweets.")
        ds = ds.where(ds[args.key_relevance] == 1, drop=True)
        logging.info(f"After selection {ds.index.shape[0]} Tweets.")
    return ds


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
    a2.utils.argparse.dataset_split(parser)
    a2.utils.argparse.dataset_select(parser)
    a2.utils.argparse.dataset_relevance(parser)
    a2.utils.argparse.dataset_weather_stations(parser)
    a2.utils.argparse.classifier(parser)
    a2.utils.argparse.dataset_rain(parser)
    a2.utils.argparse.hyperparameter_basic(parser)
    a2.utils.argparse.tracking(parser)
    a2.utils.argparse.output(parser)
    a2.utils.argparse.figures(parser)
    a2.utils.argparse.debug(parser)
    args = parser.parse_args()
    main(args)
