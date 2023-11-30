import argparse
import logging
import os

import a2.dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.model_configs
import a2.training.tracking
import a2.training.tracking_hugging
import a2.training.training_deep500
import a2.training.training_hugging
import a2.utils.argparse
import utils_scripts
from a2.training import benchmarks as timer

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main(args):
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    memory_tracker = a2.training.benchmarks.CudaMemoryMonitor()
    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
    if args.debug:
        logger.info("RUNNING IN DEBUG MODE!")
    logger.info(f"Running finetuning as {args.job_id=}")
    logger.info(f"Iteration: {args.iteration=}")
    logger.info(f"Args used: {args.__dict__}")
    model_name = os.path.split(args.model_path)[1]
    logger.info(f"Using ML model: {model_name}")

    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(args.model_path)

    path_output = utils_scripts._determine_path_output(args)
    path_figures = os.path.join(path_output, args.figure_folder)

    ds_train = utils_scripts.get_dataset(
        args,
        f"{args.filename_dataset_train}",
        path_figures=path_figures,
        tracker=tracker,
        prefix_histogram="train",
    )
    dataset_train = utils_scripts.to_hugging_face_dataset(args=args, ds=ds_train, dataset_object=dataset_object)
    ds_validate = utils_scripts.get_dataset(
        args,
        f"{args.filename_dataset_validate}",
        path_figures=path_figures,
        tracker=tracker,
        prefix_histogram="validate",
    )
    dataset_validate = utils_scripts.to_hugging_face_dataset(args=args, ds=ds_validate, dataset_object=dataset_object)
    ds_test = utils_scripts.get_dataset(
        args,
        f"{args.filename_dataset_test}",
        path_figures=path_figures,
        tracker=tracker,
        prefix_histogram="test",
    )
    dataset_test = utils_scripts.to_hugging_face_dataset(args=args, ds=ds_test, dataset_object=dataset_object)

    hyper_parameters = a2.training.model_configs.get_model_config(
        model_name=args.model_name, parameters_overwrite=args.__dict__
    )
    print(f"{hyper_parameters=}")
    trainer_object = a2.training.training_hugging.HuggingFaceTrainerClass(args.model_path, num_labels=args.num_labels)

    print(f"{dataset_train['label']=}")
    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()

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
    trainer_class = a2.training.model_configs.get_customized_trainer_class(
        args.trainer_name,
        method_overwrites=[
            args.loss,
        ],
    )
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
        trainer_class=trainer_class,
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

    # DATA
    parser.add_argument(
        "--filename_dataset_train", required=True, type=str, help="Filename of dataset used for training."
    )
    parser.add_argument(
        "--filename_dataset_validate", required=True, type=str, help="Filename of dataset used for validation."
    )
    parser.add_argument(
        "--filename_dataset_test", required=True, type=str, help="Filename of dataset used for testing."
    )
    parser.add_argument(
        "--key_input",
        type=str,
        default="text",
        help="Key that is used for input values.",
    )
    parser.add_argument(
        "--key_output",
        type=str,
        default="raining",
        help="Key that is used for output values.",
    )
    parser.add_argument(
        "--key_precipitation",
        type=str,
        default="tp_h_mm",
        help="Key that specifies precipitation level in dataset.",
    )
    parser.add_argument(
        "--key_text",
        type=str,
        default="text_normalized",
        help="Key that specifies texts used in dataset.",
    )
    parser.add_argument(
        "--precipitation_threshold_rain",
        type=float,
        default=6e-3,
        help="Values equal or higher are considered rain.",
    )

    # MODEL
    parser.add_argument(
        "--model_path",
        "-in",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/deberta-v3-small/",
        help="Directory where input netCDF-files are stored.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=a2.training.model_configs.SUPPORTED_MODELS,
        default="deberta_small",
        help="Name of model, sets default hyper parameters.",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels in classification task.",
    )
    parser.add_argument(
        "--trainer_name",
        type=str,
        default="default",
        choices=a2.training.model_configs.SUPPORTED_TRAINERS,
        help="Trainer class selected by its name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/data/training_sets_rain_classifier/dataset_split_thresh6M3/",
        help="Path to model.",
    )
    parser.add_argument(
        "--figure_folder",
        type=str,
        default="figures/",
        help="Relative path to `output_dir` for saving figures.",
    )

    # HYPERPARAMETERS
    parser.add_argument("--random_seed", "-rs", type=int, default=42, help="Random seed value.")
    parser.add_argument("--epochs", type=int, default=1, help="Numer of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per mini-batch.")
    parser.add_argument("--learning_rate", type=float, default=3e-05, help="Learning rate to train model.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay rate to train model.")
    parser.add_argument("--warmup_ratio", type=float, default=0, help="Warmup ratio to train model.")
    parser.add_argument("--warmup_steps", type=float, default=500, help="Warmup steps to train model.")
    parser.add_argument(
        "--hidden_dropout_prob", "-hdp", type=float, default=0.1, help="Probability of hidden droup out layer."
    )
    parser.add_argument("--cls_dropout", "-cd", type=float, default=0.1, help="Dropout probability in classifier head.")
    parser.add_argument("--lr_scheduler_type", "-lst", type=str, default="linear", help="Learning rate scheduler type.")

    parser.add_argument(
        "--base_model_weights_fixed",
        "-weightsfixed",
        action="store_true",
        help="Weights of base model (without classification head) are fixed (not trainable).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="default_loss",
        choices=a2.training.model_configs.SUPPORTED_LOSSES,
        help="Loss used for training selected by its name.",
    )

    # DEBUG / LOGGING
    parser.add_argument("--save_steps", type=int, default=500, help="Steps after which model is saved.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Steps after which model logs are written.")
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
    parser.add_argument(
        "--log_gpu_memory",
        action="store_true",
        help="Monitor Cuda memory usage.",
    )
    parser.add_argument(
        "--ignore_tracking",
        action="store_true",
        help="Do not use mantik tracking.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to toggle debug mode.",
    )

    parser.add_argument("--job_id", "-jid", type=int, default=None, help="Job id when running on hpc.")
    parser.add_argument("--iteration", "-i", type=int, default=0, help="Iteration number when running benchmarks.")
    args = parser.parse_args()
    main(args)