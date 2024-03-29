import argparse
import logging
import os

import a2.dataset
import a2.plotting
import a2.training.benchmarks
import a2.utils
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from a2.training import benchmarks as timer

# from deep500.utils import timer_torch as timer

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def main(args):
    memory_tracker = a2.training.benchmarks.CudaMemoryMonitor()
    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()
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
    logging.info(f"Using {len(indices_test)} for evaluation and {len(indices_train)} for training.")
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
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    experiment_id = tracker.create_experiment(args.mlflow_experiment_name)
    tracker.end_run()
    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()
    with tracker.start_run(run_name=args.run_name, experiment_id=experiment_id):
        tmr = timer.Timer()
        tracker.log_params(
            {
                "data_description": args.data_description,
                "N_tweets": N_tweets,
                "model_name": model_name,
            }
        )
        tmr.start(timer.TimeType.RUN)
        if args.log_gpu_memory:
            memory_tracker.get_cuda_memory_usage("Starting run")
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
        tracker.log_params(args.__dict__)
        tmr.start(timer.TimeType.TRAINING)
        trainer.train()
        tmr.end(timer.TimeType.TRAINING)
        tmr.start(timer.TimeType.EVALUATION)
        test_ds = dataset_object.build(ds_raw, indices_train, indices_test, train=False)
        (
            predictions,
            prediction_probabilities,
        ) = a2.training.evaluate_hugging.predict_dataset(test_ds, trainer)
        if args.log_gpu_memory:
            memory_tracker.get_cuda_memory_usage("Finished training")
        tmr.end(timer.TimeType.EVALUATION)
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
        if args.log_gpu_memory:
            memory_tracker.get_cuda_memory_usage("Finished run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    a2.utils.argparse.dataset(parser)
    a2.utils.argparse.model(parser)
    a2.utils.argparse.mlflow(parser)
    a2.utils.argparse.hyperparameter(parser)
    a2.utils.argparse.benchmarks(parser)
    args = parser.parse_args()
    main(args)
