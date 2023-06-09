import logging
import os

import a2.dataset.load_dataset
import a2.dataset.utils_dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.training_hugging
import a2.utils.argparse
import utils_scripts

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def main(args):
    """Split labeled Tweets into train, test, validation set"""
    path_output = utils_scripts._determine_path_output(args)

    _, ds_raw = utils_scripts.get_dataset(
        args, args.filename_dataset_to_split, dataset_object=None, set_labels=True, build_hugging_dataset=False
    )

    ds_selected = select_dataset(args, ds_raw)

    logging.info(f"Dataset contains {ds_selected.index.shape[0]=} Tweets.")

    indices_train, indices_validate, indices_test = a2.training.training_hugging.split_training_set_tripple(
        ds=ds_selected,
        key_stratify=args.key_stratify,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )
    ds = a2.dataset.load_dataset.reset_index_coordinate(ds_selected)

    logging.info(
        f"Using {len(indices_validate)} for evaluation, "
        f"{len(indices_test)} for testing and "
        f"{len(indices_train)} for training."
    )

    for suffix, inidices in zip(["train", "validate", "test"], [indices_train, indices_validate, indices_test]):
        save_subsection_dataset(
            ds=ds,
            filename=f"{path_output}/{args.filename_dataset_to_split.stem}_{suffix}.nc",
            indices=inidices,
        )
    path_figures = os.path.join(path_output, args.figure_folder)
    for suffix, inidices in zip(["train", "validate", "test"], [indices_train, indices_validate, indices_test]):
        utils_scripts.plot_and_log_histogram(
            ds.sel(index=inidices), "raining", path_figures, tracker=None, filename=f"raining_histogram_{suffix}.pdf"
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
    a2.utils.argparse.dataset_split(parser)
    a2.utils.argparse.dataset_select(parser)
    a2.utils.argparse.classifier(parser)
    a2.utils.argparse.dataset_rain(parser)
    a2.utils.argparse.dataset_relevance(parser)
    a2.utils.argparse.hyperparameter_basic(parser)
    a2.utils.argparse.output(parser)
    a2.utils.argparse.figures(parser)
    a2.utils.argparse.debug(parser)
    args = parser.parse_args()
    main(args)
