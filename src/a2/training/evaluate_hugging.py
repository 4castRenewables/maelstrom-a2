import logging
import typing as t
import warnings

import a2.training.dataset_hugging
import a2.training.training_hugging
import a2.training.utils_training
import datasets
import numpy as np
import torch
import transformers
import xarray


def build_ds_test(
    ds: xarray.Dataset,
    indices_test: np.ndarray,
    predictions: np.ndarray,
    prediction_probabilities: np.ndarray,
) -> xarray.Dataset:
    """
    Construct test dataset where test data is defined by their indices
    and corresponding predictions and prediction_probabilities

    Parameters:
    ----------
    ds: xarray dataset
    indices_test: Variable 'index' values that are used for testing
    predictions: Predicted labels
    prediction_probabilities: Prediction probability for both labels,
                              shape = [n_tests, 2]

    Returns
    -------
    xarray.Dataset
    """
    ds_test = ds.sel(index=indices_test)
    ds_test["prediction"] = (["index"], predictions)
    ds_test["prediction_probability_not_raining"] = (
        ["index"],
        prediction_probabilities[:, 0],
    )
    ds_test["prediction_probability_raining"] = (
        ["index"],
        prediction_probabilities[:, 1],
    )
    return ds_test


def predict_dataset(
    dataset: t.Union[datasets.Dataset, datasets.DatasetDict],
    trainer: transformers.Trainer,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on Hugging Face dataset based
    on Hugging Face trainer object

    Parameters:
    ----------
    dataset: Hugging Face dataset
    trainer: Hugging Face trainer

    Returns
    -------
    xarray.Dataset
    """
    prediction_probabilities = torch.nn.functional.softmax(
        torch.Tensor(trainer.predict(dataset).predictions), dim=1
    ).numpy()
    predictions = prediction_probabilities.argmax(-1)
    return predictions, prediction_probabilities


def make_predictions_loaded_model(
    ds: xarray.Dataset,
    indices_validate: np.ndarray,
    folder_model: str,
    key_inputs: str = "text",
    key_label: str = "raining",
    folder_tokenizer: t.Optional[transformers.DebertaTokenizer] = None,
    fp16: bool = True,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions on loaded trained Hugging Face model

    WARNING: Try using original folder for `folder_tokenizer`
             if run into issues when initializing tokenizer!
    Parameters:
    ----------
    ds: Xarray dataset
    indices_validate: Variable 'index' values that are used for validation
    folder_model: Folder of pretrained model
    key_inputs: Key of variable used as input to model
    key_label: Key of variable used as label for training
    folder_tokenizer: Folder of original model downloaded from Huuging Face
                      (may run into issues if use folder_tokenizer=folder_model)
    fp16: Whether to use fp16 16-bit (mixed) precision training
          instead of 32-bit training.

    Returns
    -------
    truth, predictions, prediction_probabilities
    """
    if not a2.training.utils_training.gpu_available():
        fp16 = False
        warnings.warn(f"16-bit evaluation only available on systems with GPU. Setting {fp16=}")
    if folder_tokenizer is None:
        folder_tokenizer = folder_model
    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(folder_tokenizer)
    dataset = dataset_object.build(
        ds,
        np.ndarray([]),
        indices_validate,
        train=False,
        key_inputs=key_inputs,
        key_label=key_label,
    )
    trainer_object_loaded = a2.training.training_hugging.HuggingFaceTrainerClass(folder_model)
    trainer = trainer_object_loaded.get_trainer(
        dataset,
        tokenizer=dataset_object.tokenizer,
        folder_output="./",
        hyper_tuning=False,
        fp16=fp16,
        evaluate=True,
    )
    logging.info("predicting")
    with torch.no_grad():
        predictions, prediction_probabilities = predict_dataset(dataset, trainer)
    logging.info("done predicting")
    truth = ds.sel(index=indices_validate)[key_label].values
    return truth, predictions, prediction_probabilities
