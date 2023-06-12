import argparse
import logging
import os

import a2.dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.tracking
import a2.training.tracking_hugging
import a2.training.training_deep500
import a2.training.training_hugging
import a2.utils
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
    model_name = os.path.split(args.model_path)[1]
    logging.info(f"Using ML model: {model_name}")

    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(args.model_path)

    path_output = utils_scripts._determine_path_output(args)
    dataset_train, _ = utils_scripts.get_dataset(
        args, f"{args.filename_dataset_train}", dataset_object, set_labels=False
    )
    dataset_validate, _ = utils_scripts.get_dataset(
        args, f"{args.filename_dataset_validate}", dataset_object, set_labels=False
    )
    dataset_test, ds_test = utils_scripts.get_dataset(
        args, f"{args.filename_dataset_test}", dataset_object, set_labels=False
    )

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
                "N_tweets_train": len(dataset_train["index"]),
                "N_tweets_validate": len(dataset_validate["index"]),
                "N_tweets_test": len(dataset_test["index"]),
                "model_name": model_name,
            }
        )
        tmr.start(timer.TimeType.RUN)
        if args.log_gpu_memory:
            memory_tracker.get_cuda_memory_usage("Starting run")
        trainer = trainer_object.get_trainer(
            (dataset_train, dataset_validate),
            hyper_parameters=hyper_parameters,
            tokenizer=dataset_object.tokenizer,
            folder_output=path_output,
            hyper_tuning=False,
            disable_tqdm=True,
            fp16=True if a2.training.utils_training.gpu_available() else False,
            callbacks=[
                a2.training.training_deep500.TimerCallback(tmr, gpu=True),
                a2.training.tracking_hugging.LogCallback(tracker),
            ],
            trainer_class=a2.training.training_deep500.TrainerWithTimer,
            base_model_trainable=not args.base_model_weights_fixed,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            evaluation_strategy=args.evaluation_strategy,
            label=args.key_output,
        )
        tracker.log_params(trainer_object.hyper_parameters.__dict__)
        tracker.log_params(args.__dict__)
        tmr.start(timer.TimeType.TRAINING)
        trainer.train()
        tmr.end(timer.TimeType.TRAINING)
        tmr.start(timer.TimeType.EVALUATION)
        (
            predictions,
            prediction_probabilities,
        ) = a2.training.evaluate_hugging.predict_dataset(dataset_test, trainer)
        if args.log_gpu_memory:
            memory_tracker.get_cuda_memory_usage("Finished training")
        tmr.end(timer.TimeType.EVALUATION)
        tmr.end(timer.TimeType.RUN)

        tmr.print_all_time_stats()
        ds_test_predicted = a2.training.evaluate_hugging.build_ds_test(
            ds=ds_test,
            indices_test=None,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            label=args.key_output,
        )
        utils_scripts.evaluate_model(
            args,
            ds_test_predicted,
            predictions,
            prediction_probabilities,
            path_figures,
            tracker,
            memory_tracker,
            hyper_parameters=hyper_parameters,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    a2.utils.argparse.dataset_relevance(parser)
    a2.utils.argparse.dataset_training(parser)
    a2.utils.argparse.classifier(parser)
    a2.utils.argparse.mlflow(parser)
    a2.utils.argparse.hyperparameter(parser)
    a2.utils.argparse.benchmarks(parser)
    a2.utils.argparse.model(parser)
    a2.utils.argparse.evaluation(parser)
    a2.utils.argparse.output(parser)
    a2.utils.argparse.debug(parser)
    args = parser.parse_args()
    main(args)
