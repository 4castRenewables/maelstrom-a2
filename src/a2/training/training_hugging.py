import dataclasses
import functools
import logging
import typing as t

import a2.dataset
import a2.plotting.analysis
import a2.training.tracking
import datasets
import numpy as np
import sklearn.model_selection
import transformers
import xarray


@dataclasses.dataclass
class HyperParametersDebertaClassifier:
    """
    Hold hyper parameters for Hugging Face DeBERTa classifier
    """

    learning_rate: float = 3e-05
    batch_size: int = 32
    weight_decay: float = 0.01
    epochs: int = 1
    warmup_ratio: float = 0  # 0.4480456499617466
    warmup_steps: float = 500  # 0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    cls_dropout: float = 0.1
    lr_scheduler_type: str = "linear"


def _compute_metrics(eval_pred):
    """
    Adopt Hugging Face metric output to prefered metric
    (f1-score) for this project

    Parameters:
    ----------
    eval_pred: Predictions and labels of Hugging Face model while training

    Returns
    -------
    dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    classification_report = a2.plotting.analysis.classification_report(
        labels,
        predictions,
        target_names=["not raining", "raining"],
        output_dict=True,
    )
    f1_weighted_average = classification_report["weighted avg"]["f1-score"]
    f1_macro_average = classification_report["macro avg"]["f1-score"]
    f1_not_raining = classification_report["not raining"]["f1-score"]
    f1_raining = classification_report["raining"]["f1-score"]
    return {
        "f1_not_raining": f1_not_raining,
        "f1_raining": f1_raining,
        "f1_weighted_average": f1_weighted_average,
        "f1_macro_average": f1_macro_average,
    }


class HuggingFaceTrainerClass:
    """
    Used to train Hugging Face models
    """

    def __init__(
        self,
        model_folder: str,
        num_labels: int = 2,
        config: t.Optional[t.Dict] = None,
    ):
        self.model_folder = model_folder
        self.num_labels = num_labels
        if config is None:
            self.db_config_base = transformers.AutoConfig.from_pretrained(model_folder, num_labels=num_labels)
        else:
            self.db_config_base = config
        self.hyper_parameters = HyperParametersDebertaClassifier()

    def get_model(self, params: t.Dict, mantik: bool = True, base_model_trainable: bool = True):
        db_config = self.db_config_base
        if params is not None:
            db_config.update({"cls_dropout": params["cls_dropout"]})
        db_config.update({"num_labels": self.num_labels})
        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_folder, config=db_config)
        if not base_model_trainable:
            for param in model.base_model.parameters():
                param.requires_grad = False
        if mantik:
            a2.training.tracking.initialize_mantik()
        return model

    def get_trainer(
        self,
        dataset: t.Union[datasets.Dataset, datasets.DatasetDict],
        hyper_parameters: HyperParametersDebertaClassifier = None,
        tokenizer: t.Optional[transformers.DebertaTokenizer] = None,
        folder_output: str = "output/",
        hyper_tuning: bool = False,
        fp16: bool = True,
        evaluate: bool = False,
        mantik: bool = True,
        disable_tqdm: bool = False,
        callbacks: list | None = None,
        base_model_trainable: bool = True,
        trainer_class=transformers.Trainer,
        logging_steps: int = 500,
        evaluation_strategy: str = "steps",
        eval_steps: int | None = 100,
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
    ):
        """
        Returns Hugging Face trainer object

        Parameters:
        ----------
        dataset: Dataset in Hugging Face format
        hyper_parameters: hyper parameters in the form of data
                          class `HyperParametersDebertaClassifier`
        tokenizer: Hugging Face tokenizer
        folder_output: Folder to save training outputs
        hyper_tuning: Whether trainer used for hyper tuning
        fp16: Whether to use fp16 16-bit (mixed) precision training
              instead of 32-bit training.
        evaluate: Whether trainer only used for evaluation
        mantik: Whether using mantik for tracking
        logging_steps: Number of steps before hugging face prints
        evaluation_strategy: When to evaluate, after "steps" or "epoch"
        eval_steps: Number of steps between evaluation (only used if `evaluation_strategy`="steps")
        save_strategy: When to save best model "steps" or "epoch"
        load_best_model_at_end: Load best model at end of training

        Returns
        -------
        Hugging Face Trainer
        """
        if hyper_parameters is None:
            hyper_parameters = HyperParametersDebertaClassifier()
        self.hyper_parameters = hyper_parameters
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_folder)
        if load_best_model_at_end and save_strategy != evaluation_strategy:
            logging.info(
                f"Setting {save_strategy=} equal to {evaluation_strategy=} "
                f"(required when {load_best_model_at_end=} = True)"
            )
            save_strategy = evaluation_strategy
        if not hyper_tuning:
            args = transformers.TrainingArguments(
                folder_output,
                learning_rate=hyper_parameters.learning_rate,
                warmup_ratio=hyper_parameters.warmup_ratio,
                warmup_steps=hyper_parameters.warmup_steps,
                lr_scheduler_type=hyper_parameters.lr_scheduler_type,
                disable_tqdm=disable_tqdm,
                fp16=fp16,
                evaluation_strategy=evaluation_strategy,
                per_device_train_batch_size=hyper_parameters.batch_size,
                per_device_eval_batch_size=hyper_parameters.batch_size,
                num_train_epochs=hyper_parameters.epochs,
                weight_decay=hyper_parameters.weight_decay,
                report_to=None,
                eval_steps=eval_steps,
                save_strategy=save_strategy,
                load_best_model_at_end=load_best_model_at_end,
                logging_steps=logging_steps,
            )
        else:
            args = transformers.TrainingArguments(
                folder_output,
                disable_tqdm=True,
                fp16=fp16,
                evaluation_strategy=evaluation_strategy,
                report_to=None,
                save_strategy=save_strategy,
                load_best_model_at_end=load_best_model_at_end,
            )
        model_init = functools.partial(self.get_model, mantik=mantik, base_model_trainable=base_model_trainable)
        if evaluate:
            return trainer_class(
                model_init=model_init,
                args=args,
                tokenizer=tokenizer,
                compute_metrics=_compute_metrics,
            )
        return trainer_class(
            model_init=model_init,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics,
            callbacks=callbacks,
        )


def split_training_set(
    ds: xarray.Dataset,
    key_stratify: str = "raining",
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
):
    """
    Returns indices of training and validation set

    Parameters:
    ----------
    ds: Xarray dataset
    key_stratify: Stratified based on this key, `None` if not required
    test_size: Fraction of validation set [0-1]
    random_state: Random seed to initialize selection
    shuffle: Whether or not to shuffle the data before splitting.
             If shuffle=False then stratify must be None.

    Returns
    -------
    Indices of training and test set
    """
    if key_stratify is None:
        stratify = ds[key_stratify].values
    else:
        stratify = None
    indices_train, indices_validate = sklearn.model_selection.train_test_split(
        np.arange(ds[a2.dataset.utils_dataset.get_variable_name_first(ds)].shape[0]),
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )
    return indices_train, indices_validate
