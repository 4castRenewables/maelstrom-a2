import typing as t

import a2.dataset.load_dataset
import datasets
import numpy as np
import transformers
import xarray


class DatasetHuggingFace:
    """
    Used to create dataset in Hugging Face format.

    Attributes
    ----------
    tokenizer : Hugging Face tokenizer

    Notes
    -----
    Use Hugging Face model folders to initialize tokenizer and build a
    Hugging Face dataset from an xarray dataset.
    Can be reused to bild the test dataset for validation.
    """

    def __init__(self, model_folder: str, use_fast: bool = False):
        self.model_folder = model_folder
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_folder,
            use_fast=False,
        )

    def _tok_func(self, x: dict):
        return self.tokenizer(x["inputs"], padding=True)

    def build(
        self,
        ds: xarray.Dataset,
        indices_train: np.ndarray | None,
        indices_validate: np.ndarray | None,
        train: bool = True,
        key_inputs: str = "text",
        key_label: str = "raining",
        reset_index: bool = True,
        prediction_dataset: bool = False,
    ) -> t.Union[datasets.Dataset, datasets.DatasetDict]:
        """
        Create Hugging Face dataset (datasets.DatasetDict) from xarray dataset

        Parameters:
        ----------
        ds: xarray dataset
        indices_train: Variable 'index' values that are used for training
        indices_validate: Variable 'index' values that are used for validation
        train: If the dataset is used for training
        key_inputs: Key of variable used as input to model
        key_label: Key of variable used as label for training
        reset_index: Reset index coordinate
        prediction_dataset: Whether to build a dataset used for prediction (no labels required, only added if present).

        Returns
        -------
        datasets.DatasetDict
        """
        if train and (sum([indices_train is None, indices_validate is None]) == 1):
            raise ValueError(f"{indices_train=} and {indices_validate} can either be both None or have to be set!")
        if prediction_dataset:
            required_keys = [key_inputs]
            if key_label in ds:
                required_keys = required_keys + [key_label]
        else:
            required_keys = [key_inputs, key_label]
        a2.dataset.utils_dataset.assert_keys_in_dataset(ds, required_keys)
        if reset_index:
            ds = a2.dataset.load_dataset.reset_index_coordinate(ds)
        if not train and indices_validate is not None:
            ds = ds.sel(index=indices_validate)
        df = ds[required_keys].to_pandas()
        columns: t.Mapping = {key_inputs: "inputs", key_label: "label"}
        df = df.rename(columns=columns, errors="ignore")  # type: ignore
        datasets_ds = datasets.Dataset.from_pandas(df)
        if key_label in required_keys:
            updated_features = datasets_ds.features.copy()
            updated_features["label"] = datasets.ClassLabel(names=[f"not {key_label}", f"{key_label}"])
            datasets_ds = datasets_ds.cast(updated_features)
        tok_ds = datasets_ds.map(self._tok_func, batched=True)
        if not train or (indices_train is None and indices_validate is None):
            return tok_ds
        else:
            return datasets.DatasetDict(
                {
                    "train": tok_ds.select(indices_train),
                    "test": tok_ds.select(indices_validate),
                }
            )
