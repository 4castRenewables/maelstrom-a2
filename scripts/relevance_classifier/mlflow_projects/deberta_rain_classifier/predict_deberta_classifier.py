import argparse
import logging
import os

import a2.dataset
import a2.plotting.histograms
import a2.training.benchmarks
import a2.training.evaluate_hugging
import a2.training.tracking
import a2.training.tracking_hugging
import a2.training.training_deep500
import a2.utils.argparse
import torch
import utils_scripts
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
    if args.debug:
        logging.info("RUNNING IN DEBUG MODE!")
    logging.info(f"Running finetuning as {args.job_id=}")
    logging.info(f"Iteration: {args.iteration=}")
    logging.info(f"Args used: {args.__dict__}")
    model_name = os.path.split(args.path_trained_model)
    logging.info(f"Using ML model: {model_name}")

    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(args.path_trained_model)

    path_output = utils_scripts._determine_path_output(args)
    dataset_predict, ds_predict = utils_scripts.get_dataset(
        args, args.filename_dataset_predict, dataset_object, set_labels=False
    )

    path_figures = os.path.join(path_output, args.figure_folder)
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    experiment_id = tracker.create_experiment(args.mlflow_experiment_name)
    tracker.end_run()
    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()
    with tracker.start_run(run_name=args.run_name, experiment_id=experiment_id):
        tmr = timer.Timer()
        tracker.log_params(
            {
                "N_tweets_predict": len(dataset_predict["index"]),
                "model_name": model_name,
            }
        )
        tmr.start(timer.TimeType.RUN)
        if args.log_gpu_memory:
            memory_tracker.get_cuda_memory_usage("Starting run")
        trainer_object_loaded = a2.training.training_hugging.HuggingFaceTrainerClass(args.path_trained_model)
        trainer = trainer_object_loaded.get_trainer(
            dataset_predict,
            tokenizer=dataset_object.tokenizer,
            folder_output=path_output,
            hyper_tuning=False,
            fp16=True if a2.training.utils_training.gpu_available() else False,
            evaluate=True,
        )
        tmr.start(timer.TimeType.EVALUATION)
        with torch.no_grad():
            predictions, prediction_probabilities = a2.training.evaluate_hugging.predict_dataset(
                dataset_predict, trainer
            )

        if args.log_gpu_memory:
            memory_tracker.get_cuda_memory_usage("Finished training")
        tmr.end(timer.TimeType.EVALUATION)
        tmr.end(timer.TimeType.RUN)
        tmr.print_all_time_stats()

        ds_test_predicted = a2.training.evaluate_hugging.build_ds_test(
            ds=ds_predict,
            indices_test=None,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            label=args.key_output,
        )
        a2.dataset.load_dataset.save_dataset(
            ds_test_predicted,
            filename=f"{path_output}/{args.filename_dataset_predict.stem}_{args.key_output}_predicted.nc",
        )

        # only "evaluate" prediction if true label present in dataset
        if args.key_output in ds_test_predicted:
            utils_scripts.evaluate_model(
                args,
                ds_test_predicted,
                predictions,
                prediction_probabilities,
                path_figures,
                tracker,
                memory_tracker,
            )
        utils_scripts.plot_and_log_histogram(
            ds_test_predicted,
            f"prediction_{args.key_output}",
            path_figures,
            tracker=tracker,
            filename="prediction_histogram.pdf",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    a2.utils.argparse.classifier(parser)
    a2.utils.argparse.hyperparameter_basic(parser)
    a2.utils.argparse.predict_model(parser)
    a2.utils.argparse.mlflow(parser)
    a2.utils.argparse.benchmarks(parser)
    a2.utils.argparse.model(parser)
    a2.utils.argparse.output(parser)
    a2.utils.argparse.debug(parser)
    args = parser.parse_args()
    main(args)
