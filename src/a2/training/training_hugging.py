import dataclasses
import functools
import logging
import typing as t

import a2.dataset
import a2.plotting.analysis
import a2.training.tracking
import a2.training.utils_training
import a2.utils.utils
import datasets
import numpy as np
import sklearn.model_selection
import transformers

xarray, xarray_dataset_type = a2.utils.utils._import_xarray_and_define_xarray_type(__file__)

torch = a2.utils.utils._import_torch(__file__)


@dataclasses.dataclass
class HyperParametersHuggingFaceClassifier:
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

    def update(self, new: dict):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclasses.dataclass
class HyperParametersDebertaClassifier(HyperParametersHuggingFaceClassifier):
    """
    Hold hyper parameters for Hugging Face DeBERTa classifier
    """

    learning_rate: float = 3e-05


@dataclasses.dataclass
class HyperParametersElectraClassifier(HyperParametersHuggingFaceClassifier):
    """
    Hold hyper parameters for Hugging Face DeBERTa classifier
    """

    learning_rate: float = 2e-5


def _compute_metrics(eval_pred, label="raining"):
    """
    Adopt Hugging Face metric output to prefered metric
    (f1-score) for this project

    Parameters:
    ----------
    eval_pred: Predictions and labels of Hugging Face model while training
    label: Label name of the classification

    Returns
    -------
    dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    classification_report = a2.plotting.analysis.classification_report(
        labels,
        predictions,
        label=label,
        output_dict=True,
    )
    f1_weighted_average = classification_report["weighted avg"]["f1-score"]
    f1_macro_average = classification_report["macro avg"]["f1-score"]
    f1_not = classification_report[f"not {label}"]["f1-score"]
    f1 = classification_report[label]["f1-score"]
    return {
        f"f1_not_{label}": f1_not,
        f"f1_{label}": f1,
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
        problem_type: str = "single_label_classification",
    ):
        self.model_folder = model_folder
        self.num_labels = num_labels
        if config is None and self.num_labels is not None:
            self.db_config_base = transformers.AutoConfig.from_pretrained(
                model_folder, num_labels=num_labels, problem_type=problem_type
            )
        elif config is None:
            self.db_config_base = transformers.AutoConfig.from_pretrained(model_folder, problem_type=problem_type)
        else:
            self.db_config_base = config
        self.hyper_parameters = HyperParametersDebertaClassifier()

    def get_model(self, params: t.Dict, mantik: bool = True, base_model_trainable: bool = True):
        db_config = self.db_config_base
        if params is not None:
            db_config.update({"cls_dropout": params["cls_dropout"]})
        if self.num_labels is not None:
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
        dataset: t.Union[datasets.Dataset, datasets.DatasetDict] | None,
        hyper_parameters: HyperParametersDebertaClassifier | None = None,
        tokenizer: t.Optional[transformers.DebertaTokenizer] | None = None,
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
        save_steps: int = 500,
        evaluation_strategy: str = "steps",
        eval_steps: int | None = 100,
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        label: str = "raining",
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
        disable_tqdm: Whether to disable progress bar used by `Transformer`
        callbacks: Callbacks during training
        base_model_trainable: Whether base model weights are trainable (not fixed)
        trainer_class: Trainer class compatible with `transformer.Trainer`
        logging_steps: Number of steps before hugging face prints
        evaluation_strategy: When to evaluate, after "steps" or "epoch"
        eval_steps: Number of steps between evaluation (only used if `evaluation_strategy`="steps")
        save_strategy: When to save best model "steps" or "epoch"
        load_best_model_at_end: Load best model at end of training
        label: Label name of the classification

        Returns
        -------
        Hugging Face Trainer
        """
        if dataset is None and not evaluate:
            raise ValueError(
                f"Need to specify dataset {dataset=} for training.\nFor evaluation set `evaluat=True` ({evaluate=})!"
            )
        if not a2.training.utils_training.cuda_available():
            fp16 = False
        if hyper_parameters is None:
            hyper_parameters = HyperParametersHuggingFaceClassifier()
        self.hyper_parameters = hyper_parameters
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_folder)
        if load_best_model_at_end and save_strategy != evaluation_strategy:
            logging.info(
                f"Setting {save_strategy=} equal to {evaluation_strategy=} "
                f"(required when {load_best_model_at_end=} = True)"
            )
            save_strategy = evaluation_strategy
            eval_steps, save_steps, logging_steps = set_evaluation_save_logging_steps(
                evaluation_strategy=evaluation_strategy,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
            )
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
                save_steps=save_steps,
                save_strategy=save_strategy,
                logging_steps=logging_steps,
                load_best_model_at_end=load_best_model_at_end,
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
        _partial_compute_metrics = functools.partial(_compute_metrics, label=label)
        if evaluate:
            return trainer_class(
                model_init=model_init,
                args=args,
                tokenizer=tokenizer,
                compute_metrics=_partial_compute_metrics,
                callbacks=callbacks,
            )
        train_dataset, eval_dataset = _get_training_evaluation_datasets(dataset)
        return trainer_class(
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=_partial_compute_metrics,
            callbacks=callbacks,
        )


def _get_training_evaluation_datasets(dataset):
    if isinstance(dataset, datasets.DatasetDict):
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    elif isinstance(dataset, tuple) and len(dataset) == 2:
        train_dataset = dataset[0]
        eval_dataset = dataset[1]
    else:
        raise TypeError(f"Couldn't parse {dataset=}!")
    return train_dataset, eval_dataset


def split_training_set(
    ds: xarray_dataset_type,
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
    validation_size: Fraction of validation set; 1 - validation_size - test_size > 0
    test_size: Fraction of test set; 1 - validation_size - test_size > 0
    random_state: Random seed to initialize selection
    shuffle: Whether or not to shuffle the data before splitting.
             If shuffle=False then stratify must be None.

    Returns
    -------
    Indices of training, validation and test set
    """
    if key_stratify is not None:
        stratify = ds[key_stratify].values
    else:
        stratify = None
    indices_train, indices_test = sklearn.model_selection.train_test_split(
        np.arange(ds[a2.dataset.utils_dataset.get_variable_name_first(ds)].shape[0]),
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )
    return indices_train, indices_test


def split_training_set_tripple(
    ds: xarray_dataset_type,
    key_stratify: str = "raining",
    validation_size: float | None = None,
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
    validation_size: Fraction of validation set; 1 - validation_size - test_size > 0
    test_size: Fraction of test set; 1 - validation_size - test_size > 0
    random_state: Random seed to initialize selection
    shuffle: Whether or not to shuffle the data before splitting.
             If shuffle=False then stratify must be None.

    Returns
    -------
    Indices of training, validation and test set
    """
    ds = a2.dataset.load_dataset.reset_index_coordinate(ds.copy())
    train_size = 1 - test_size
    if validation_size is not None:
        train_size -= validation_size
    if train_size < 0:
        raise ValueError(f"{train_size=} is below zero! (Decrease {validation_size=} and/or {test_size=})")
    indices_train, indices_test = split_training_set(
        ds=ds,
        key_stratify=key_stratify,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    indices_validate = np.array([], dtype=int)
    if validation_size is not None:
        indices_train, indices_validate = split_training_set(
            ds=a2.dataset.utils_dataset.select_rows_by_index(ds, indices=indices_train),
            key_stratify=key_stratify,
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
            shuffle=shuffle,
        )
    return indices_train, indices_validate, indices_test


def set_evaluation_save_logging_steps(
    evaluation_strategy: str, eval_steps: int = 500, logging_steps: None | int = None, save_steps: None | int = None
):
    logging.info(f"User choice: {eval_steps=} {logging_steps=} {save_steps=}, ({evaluation_strategy=})")
    if eval_steps is None:
        raise ValueError(f"{eval_steps=} cannot be None!")
    if evaluation_strategy == "epoch":
        # eval_steps is ignored
        pass
    elif evaluation_strategy == "steps":
        if eval_steps is not None:
            save_steps = eval_steps
    elif evaluation_strategy == "no":
        pass
    else:
        raise ValueError(f"{evaluation_strategy=} not known!")
    if logging_steps is None:
        logging_steps = eval_steps
    if save_steps is None:
        save_steps = eval_steps
    logging.info(f"Setting: {eval_steps=} {logging_steps=} {save_steps=}, ({evaluation_strategy=})")
    return eval_steps, save_steps, logging_steps


class TrainerWithFocalLoss(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        HP_LOSS_FL = (0.2, 5, 3)
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        labels = labels.tile((logits.shape[0],)).unsqueeze(-1)
        print(f"{labels=}")
        print(f"{inputs=}")
        print(f"{logits=}")
        probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten().gather(1, labels).flatten()
        # probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten(0, 1).gather(1, labels).flatten()
        gammas = torch.where(probabilities < HP_LOSS_FL[0], HP_LOSS_FL[1], HP_LOSS_FL[2])
        loss = -((1 - probabilities) ** gammas * probabilities.log()).mean()
        return (loss, outputs) if return_outputs else loss
