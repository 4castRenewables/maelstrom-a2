import argparse
import logging
import os

import a2.dataset
import a2.plotting
import a2.training
import a2.utils
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from a2.training import benchmarks as timer

# from deep500.utils import timer_torch as timer

logging.basicConfig(level=logging.INFO)


def main(args):
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
    logging.info(f"Running finetuning as {args.job_id=}")
    logging.info(f"Iteration: {args.iteration=}")
    logging.info(f"Args used: {args.__dict__}")
    ds_raw = a2.dataset.load_dataset.load_tweets_dataset(os.path.join(args.data_dir, args.data_filename), raw=True)
    N_tweets = ds_raw.index.shape[0]
    logging.info(f"loaded {N_tweets} tweets")
    ds_raw["text"] = (["index"], ds_raw[args.key_text].values.copy())
    ds_raw["raining"] = (["index"], np.array(ds_raw[args.key_rain].values > args.threshold_rain, dtype=int))

    indices_train, indices_test = sklearn.model_selection.train_test_split(
        np.arange(ds_raw["index"].shape[0]),
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
        stratify=ds_raw.raining.values,
    )
    model_name = os.path.split(args.model_path)[1]
    logging.info(f"Using ML model: {model_name}")
    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(args.model_path)
    dataset = dataset_object.build(ds_raw, indices_train, indices_test)

    hyper_parameters = a2.training.training_hugging.HyperParametersDebertaClassifier(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.number_epochs,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        hidden_dropout_prob=args.hidden_dropout_prob,
        cls_dropout=args.cls_dropout,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    trainer_object = a2.training.training_hugging.HuggingFaceTrainerClass(args.model_path)

    path_run = os.path.join(args.output_dir, args.run_folder)
    path_figures = os.path.join(path_run, args.figure_folder)
    tracker = a2.training.tracking.Tracker()
    tracker.end_run()
    a2.training.tracking.initialize_mantik()
    with tracker.start_run(run_name=args.run_name):
        tmr = timer.Timer()
        a2.training.tracking.initialize_mantik()
        tracker.log_params(
            {
                "data_description": args.data_description,
                "N_tweets": N_tweets,
                "model_name": model_name,
            }
        )
        tmr.start(timer.TimeType.RUN)
        trainer = trainer_object.get_trainer(
            dataset,
            hyper_parameters=hyper_parameters,
            tokenizer=dataset_object.tokenizer,
            folder_output=path_run,
            hyper_tuning=False,
            disable_tqdm=True,
            fp16=True if a2.training.utils_training.gpu_available() else False,
            callbacks=[
                a2.training.training_deep500.TimerCallback(tmr, gpu=True),
                a2.training.tracking_hugging.LogCallback(tracker),
            ],
            trainer_class=a2.training.training_deep500.TrainerWithTimer,
            logging_steps=1,
            base_model_trainable=not args.base_model_weights_fixed,
            eval_steps=args.eval_steps,
            evaluation_strategy=args.evaluation_strategy,
        )
        tracker.log_params(trainer_object.hyper_parameters.__dict__)
        tmr.start(timer.TimeType.TRAINING)
        trainer.train()
        tmr.end(timer.TimeType.TRAINING)
        test_ds = dataset_object.build(ds_raw, indices_train, indices_test, train=False)
        (
            predictions,
            prediction_probabilities,
        ) = a2.training.evaluate_hugging.predict_dataset(test_ds, trainer)
        tmr.end(timer.TimeType.RUN)

        tmr.print_all_time_stats()
        ds_test = a2.training.evaluate_hugging.build_ds_test(
            ds=ds_raw,
            indices_test=indices_test,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
        )
        truth = ds_test.raining.values

        a2.training.tracking.log_metric_classification_report(tracker, truth, predictions, step=hyper_parameters.epochs)

        filename_certainty_plot = os.path.join(path_figures, "plot_2d_predictions_truth.pdf")
        a2.plotting.analysis.plot_prediction_certainty(
            truth=ds_test["raining"].values,
            prediction_probabilities=ds_test["prediction_probability_raining"].values,
            filename=filename_certainty_plot,
        )
        tracker.log_artifact(filename_certainty_plot)

        filename_roc_plot = os.path.join(path_figures, "roc.pdf")
        a2.plotting.analysis.plot_roc(
            ds_test.raining.values, prediction_probabilities[:, 1], filename=filename_roc_plot
        )
        tracker.log_artifact(filename_roc_plot)
        logging.info(f"Max memory consumption [Gbyte]: {timer.get_max_memory_usage()/1e9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-inpath",
        type=str,
        default=a2.utils.file_handling.get_folder_data() / "tweets/",
        help="Directory where input netCDF-files are stored.",
    )
    parser.add_argument(
        "--data_filename",
        "-infile",
        type=str,
        default="2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered.nc",
        help="Filename of training data.",
    )
    parser.add_argument(
        "--data_description",
        "-datad",
        type=str,
        default="tweets 2017-2020, keywords emojis as description, keywords only",
        help="Data description purely used for logging.",
    )

    parser.add_argument(
        "--model_path",
        "-in",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/deberta-v3-small/",
        help="Directory where input netCDF-files are stored.",
    )
    parser.add_argument(
        "--output_dir",
        "-outdir",
        type=str,
        default=a2.utils.file_handling.get_folder_models(),
        help="Output directory where model is saved.",
    )
    parser.add_argument(
        "--run_folder", type=str, required=True, help="Output folder where model is saved in `output_dir`."
    )
    parser.add_argument("--run_name", type=str, default="era5 whole dataset", help="Name of run used for logging only.")

    parser.add_argument(
        "--figure_folder", type=str, default="figures/", help="Directory where input netCDF-files are stored."
    )

    parser.add_argument("--key_text", type=str, default="text_normalized", help="Key for text in input file.")
    parser.add_argument("--key_rain", type=str, default="tp_h", help="Key for rain data in input file.")
    parser.add_argument(
        "--threshold_rain",
        type=float,
        default=1e-8,
        help="Threshold above which precipitation is considered raining in input file.",
    )

    parser.add_argument("--number_epochs", "-nepochs", type=int, default=1, help="Numer of epochs to train.")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Number of samples per mini-batch.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-05, help="Learning rate to train model.")
    parser.add_argument("--weight_decay", "-wd", type=float, default=0, help="Weight decay rate to train model.")
    parser.add_argument("--warmup_ratio", "-wr", type=float, default=0, help="Warmup ratio to train model.")
    parser.add_argument("--warmup_steps", "-ws", type=float, default=500, help="Warmup steps to train model.")
    parser.add_argument(
        "--hidden_dropout_prob", "-hdp", type=float, default=0.1, help="Probability of hidden droup out layer."
    )
    parser.add_argument("--cls_dropout", "-cd", type=float, default=0.1, help="Dropout probability in classifier head.")
    parser.add_argument("--lr_scheduler_type", "-lst", type=str, default="linear", help="Learning rate scheduler type.")

    parser.add_argument("--test_size", "-ts", type=float, default=0.2, help="Fraction of test set.")

    parser.add_argument(
        "--evaluation_strategy", "-estrat", type=str, default="epoch", help="When to evaluate steps/epoch/no"
    )
    parser.add_argument(
        "--eval_steps",
        "-esteps",
        type=int,
        default=None,
        help="Number of steps between evaluations if evaluation strategy is set to steps.",
    )

    parser.add_argument("--random_seed", "-rs", type=int, default=42, help="Random seed value.")
    parser.add_argument("--iteration", "-i", type=int, default=0, help="Iteration number when running benchmarks.")
    parser.add_argument("--job_id", "-jid", type=int, default=None, help="Job id when running on hpc.")
    parser.add_argument(
        "--base_model_weights_fixed",
        "-weightsfixed",
        action="store_true",
        help="Weights of base model (without classification head) are fixed (not trainable).",
    )

    args = parser.parse_args()
    main(args)
