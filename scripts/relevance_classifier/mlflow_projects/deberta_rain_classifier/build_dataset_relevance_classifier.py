import logging
import os

import a2.dataset.load_dataset
import a2.dataset.utils_dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.training_hugging
import a2.utils.argparse
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import utils_scripts

# from deep500.utils import timer_torch as timer

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def main(args):
    """ "Split Tweets (matched by keywords) into args.key_relevance Tweets used for relevance classifier,
    excluding Tweets near weather station (<1km),
    remaining Tweets are used for training the "raining" classifier"""
    logging.info(f"Args used: {args.__dict__}")
    dataset_prefix = _determine_dataset_prefix(args)
    path_output = utils_scripts._determine_path_output(args)

    ds_relevant_raw = _prepare_relevant_tweets(args, dataset_prefix, path_output)

    ds_irrelevant = _prepare_irrelevant_tweets(args)

    N_tweets_relevant, N_tweets_irrelevant = ds_relevant_raw.index.shape[0], ds_irrelevant.index.shape[0]
    logging.info(f"RAW: loaded {N_tweets_relevant=}, {N_tweets_irrelevant=}")

    if N_tweets_relevant <= N_tweets_irrelevant:
        raise ValueError(
            f"{N_tweets_relevant=} <= {N_tweets_irrelevant=},"
            "expected at least same amount of relevant as irrelevant Tweets"
        )
    split_raining_relevance_classifiers = N_tweets_irrelevant / N_tweets_relevant
    indices_raining_classifier, indices_relevance_classifier = sklearn.model_selection.train_test_split(
        np.arange(ds_relevant_raw["index"].shape[0]),
        test_size=split_raining_relevance_classifiers,
        random_state=args.random_seed,
        shuffle=True,
        stratify=ds_relevant_raw.raining.values,
    )
    ds_relevant_raw = a2.dataset.load_dataset.reset_index_coordinate(ds_relevant_raw)
    ds_relevant = ds_relevant_raw.sel(index=indices_relevance_classifier)

    ds_raining_classifier = ds_relevant_raw.sel(index=indices_raining_classifier)

    ds_relevant[args.key_relevance] = (["index"], np.ones(ds_relevant.index.shape[0], dtype=bool))
    ds_irrelevant[args.key_relevance] = (["index"], np.zeros(ds_irrelevant.index.shape[0], dtype=bool))
    ds_relevance_dataset = a2.dataset.utils_dataset.merge_datasets_along_index(ds_relevant, ds_irrelevant)
    ds_relevance_dataset = a2.dataset.load_dataset.reset_index_coordinate(ds_relevance_dataset)
    (
        indices_train_relevance,
        indices_validate_relevance,
        indices_test_relevance,
    ) = a2.training.training_hugging.split_training_set_tripple(
        ds_relevance_dataset,
        validation_size=args.validation_size,
        test_size=args.test_size,
        key_stratify=args.key_relevance,
        random_state=args.random_seed,
    )
    logging.info(
        f"Using {len(indices_validate_relevance)} for evaluation, "
        f"{len(indices_test_relevance)} for testing and "
        f"{len(indices_train_relevance)} for training."
    )
    a2.dataset.load_dataset.save_dataset(
        ds_raining_classifier,
        filename=f"{path_output}{args.raining_classifier_dataset_prefix}{dataset_prefix}.nc",
    )

    for suffix, inidices in zip(
        ["train", "validate", "test"], [indices_train_relevance, indices_validate_relevance, indices_test_relevance]
    ):
        save_subsection_dataset(
            ds=ds_relevance_dataset,
            filename=f"{path_output}{args.relevance_classifier_dataset_prefix}{dataset_prefix}_{suffix}.nc",
            indices=inidices,
        )


def save_subsection_dataset(ds, filename, indices):
    ds_subsection = ds.sel(index=indices)
    a2.dataset.load_dataset.save_dataset(ds_subsection, filename=filename, reset_index=True, no_conversion=False)


def _prepare_irrelevant_tweets(args):
    ds_tweets_random = a2.dataset.load_dataset.load_tweets_dataset(
        os.path.join(args.tweets_dir, args.data_filename_irrelevant),
        reset_index_raw=True,
    )
    if args.n_tweets_irrelevant != -1:
        ds_tweets_random = ds_tweets_random.sel(index=slice(int(args.n_tweets_irrelevant)))
    return ds_tweets_random


def _prepare_relevant_tweets(args, dataset_prefix, path_output):
    ds_tweets_keywords = a2.dataset.load_dataset.load_tweets_dataset(
        os.path.join(args.tweets_dir, args.filename_tweets_with_keywords)
    )

    ds_tweets_keywords_excluded = utils_scripts.exclude_and_save_weather_stations_dataset(
        args, ds_tweets_keywords, path_output, dataset_prefix
    )
    ds_tweets_keywords_excluded["raining"] = (
        ["index"],
        np.array(ds_tweets_keywords_excluded[args.key_rain].values > args.threshold_rain, dtype=int),
    )

    return ds_tweets_keywords_excluded


def _determine_dataset_prefix(args):
    dataset_prefix, _ = os.path.splitext(args.filename_tweets_with_keywords)
    logging.info(f"Determine {dataset_prefix=}")
    return dataset_prefix


def _initialize_tracking(args):
    memory_tracker = a2.training.benchmarks.CudaMemoryMonitor()
    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
    logging.info(f"Running finetuning as {args.job_id=}")
    logging.info(f"Iteration: {args.iteration=}")
    logging.info(f"Args used: {args.__dict__}")
    return memory_tracker


if __name__ == "__main__":
    parser = a2.utils.argparse.get_parser()
    a2.utils.argparse.dataset_tweets(parser)
    a2.utils.argparse.dataset_split_sizes(parser)
    a2.utils.argparse.dataset_rain(parser)
    a2.utils.argparse.dataset_relevance(parser)
    a2.utils.argparse.dataset_relevance_split(parser)
    a2.utils.argparse.hyperparameter_basic(parser)
    a2.utils.argparse.output(parser)
    args = parser.parse_args()
    main(args)
