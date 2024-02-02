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
import a2.training.model_infos
import a2.utils.argparse
import utils_scripts
from a2.training import benchmarks as timer
import numpy as np

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main(args):
    os.environ["A2_DATASET_BACKEND"] = args.dataset_backend
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    memory_tracker = a2.training.benchmarks.CudaMemoryMonitor()
    if a2.training.utils_training.cuda_available():
        logger.info(f"Running on device: {a2.training.utils_training.available_cuda_devices_names()}.")
    # if args.log_gpu_memory:
    #     memory_tracker.reset_cuda_memory_monitoring()
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
    if args.debug:
        logger.info("RUNNING IN DEBUG MODE!")
    logger.info(f"Running finetuning as {args.job_id=}")
    if "SLURM_NODELIST" in os.environ:
        slurm_nodelist = os.environ["SLURM_NODELIST"]
        logger.info(f"Running on nodes {slurm_nodelist}.")
    logger.info(f"Iteration: {args.iteration=}")
    logger.info(f"Args used: {args.__dict__}")
    model_name = os.path.split(args.model_path)[1]
    logger.info(f"Using ML model: {model_name}")
    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(args.model_path)

    path_output = utils_scripts._determine_path_output(args)
    path_figures = os.path.join(path_output, args.folder_figures)
    path_save_models = os.path.join(path_output, args.folder_saved_models)
    path_power_logs = None
    if args.log_gpu_power:
        path_power_logs = os.path.join(path_output, args.folder_power_logs)
        a2.utils.file_handling.make_directories(path_power_logs)
        logger.info(f"Creating folder to save power logs at {path_power_logs}.")

    ds_train = utils_scripts.get_dataset(
        args,
        f"{args.filename_dataset_train}",
        path_figures=path_figures,
        tracker=tracker,
        prefix_histogram="train",
        setting_rain=False,
    )
    dataset_train = utils_scripts.to_hugging_face_dataset(args=args, ds=ds_train, dataset_object=dataset_object)
    ds_validate = utils_scripts.get_dataset(
        args,
        f"{args.filename_dataset_validate}",
        path_figures=path_figures,
        tracker=tracker,
        prefix_histogram="validate",
        setting_rain=False,
    )
    dataset_validate = utils_scripts.to_hugging_face_dataset(args=args, ds=ds_validate, dataset_object=dataset_object)
    ds_test = utils_scripts.get_dataset(
        args,
        f"{args.filename_dataset_test}",
        path_figures=path_figures,
        tracker=tracker,
        prefix_histogram="test",
        setting_rain=False,  # already done in build_dataset_rain_classifier.py
    )
    dataset_test = utils_scripts.to_hugging_face_dataset(args=args, ds=ds_test, dataset_object=dataset_object)

    hyper_parameters = a2.training.model_configs.get_model_config(
        model_name=args.model_name, parameters_overwrite=args.__dict__
    )
    logger.info(f"{hyper_parameters=}")
    trainer_object = a2.training.training_hugging.HuggingFaceTrainerClass(args.model_path, num_labels=args.num_labels)
    logger.info(f"{dataset_train['label']=}")
    # if args.log_gpu_memory:
    #     memory_tracker.reset_cuda_memory_monitoring()

    tmr = timer.Timer()
    tracker.log_params(
        {
            "N_tweets_train": a2.dataset.utils_dataset.ds_shape(dataset_train),
            "N_tweets_validate": a2.dataset.utils_dataset.ds_shape(dataset_validate),
            "N_tweets_test": a2.dataset.utils_dataset.ds_shape(dataset_test),
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
        folder_output=path_save_models,
        hyper_tuning=False,
        disable_tqdm=True,
        fp16=True if a2.training.utils_training.cuda_available() else False,
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
    logger.info(f"Device map:\n{dir(trainer.model)}")
    logger.info(f"Device map:\n{trainer.model.device}")
    n_model_parameters = a2.training.model_infos.n_model_parameters(trainer.model)
    logger.info(f"{n_model_parameters=}")
    tracker.log_params(n_model_parameters)
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
    return tracker, path_power_logs


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
        "--key_raining",
        type=str,
        default="raining",
        help="Key that specifies precipitation level in dataset.",
    )
    parser.add_argument(
        "--key_text",
        type=str,
        default="text_normalized",
        help="Key that specifies texts used in dataset.",
    )
    parser.add_argument(
        "--dataset_backend",
        type=str,
        choices=a2.utils.constants.TYPE_DATASET_BACKEND,
        default="xarray",
        help="Name of backend used to process datasets, e.g. xarray, pandas.",
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
        "--log_gpu_power",
        action="store_true",
        help="Monitor Cuda power consumption on Juwels.",
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

    # OUTPUTS
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/data/training_sets_rain_classifier/",
        help="Path to model.",
    )
    parser.add_argument(
        "--folder_run",
        type=str,
        default="dataset_split_thresh6M3/",
        help="Relative path to all ouputs connected to this run.",
    )
    parser.add_argument(
        "--folder_saved_models",
        type=str,
        default="saved_model/",
        help="Relative path to trained model.",
    )
    parser.add_argument(
        "--folder_power_logs",
        type=str,
        default="power/",
        help="Relative path to power logs.",
    )
    parser.add_argument(
        "--folder_figures",
        type=str,
        default="figures/",
        help="Relative path to `output_dir` for saving figures.",
    )

    parser.add_argument("--job_id", "-jid", type=int, default=None, help="Job id when running on hpc.")
    parser.add_argument("--iteration", "-i", type=int, default=0, help="Iteration number when running benchmarks.")
    args = parser.parse_args()

    logger.info(f"Whether to log power consumption: {args.log_gpu_power=}, ({bool(args.log_gpu_power)=})")

    hardware = "nvidia" if "CUDA_VERSION" in os.environ else "amd"
    logger.info(f"Assuming ${hardware} GPU hardware")

    if hardware == "amd":
        import utils_energy_amd as utils_energy
    else:
        import utils_energy_nvidia as utils_energy

    if args.log_gpu_power:
        with utils_energy.GetPower() as measured_scope:
            logger.info("Measuring Energy during main() call")
            # try:
            tracker, path_power_logs = main(args)
        # except Exception as exc:
        #     import traceback
        #     logger.info(f"Errors occured during training: {exc}")
        #     logger.info(f"Traceback: {traceback.format_exc()}")

        logger.info("Energy data:")
        logger.info(measured_scope.df)
        logger.info("Energy-per-GPU-list:")
        energy_int = measured_scope.energy()
        logger.info(f"integrated: {energy_int}")
        filename_power_csv = utils_energy.save_energy_to_file(measured_scope, path_power_logs, args.job_id)
        power = utils_scripts.load_power(filename=filename_power_csv)
        logger.info(
            "---Power report:---\n"
            f"total_power_consumption_Wh_per_device: {'-'.join(np.array(utils_scripts.total_power_consumption_Wh_per_device(power, 8), str))}\n"  # noqa: E501
            f"total_power_consumption_Wh: {utils_scripts.total_power_consumption_Wh(power, 8)}\n"
            f"average_consumption_W_per_device: {'-'.join(np.array(utils_scripts.average_consumption_W_per_device(power), str))}\n"  # noqa: E501
        )

    else:
        logger.info("Not measuring energy")
        main(args)
