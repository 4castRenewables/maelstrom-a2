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
        indices_train: np.ndarray,
        indices_validate: np.ndarray,
        train: bool = True,
        key_inputs: str = "text",
        key_label: str = "raining",
        reset_index: bool = True,
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

        Returns
        -------
        datasets.DatasetDict
        """
        a2.dataset.utils_dataset.assert_keys_in_dataset(ds, [key_inputs, key_label])
        if reset_index:
            ds = a2.dataset.load_dataset.reset_index_coordinate(ds)
        if not train:
            ds = ds.sel(index=indices_validate)
        df = ds[[key_inputs, key_label]].to_pandas()
        columns: t.Mapping = {key_inputs: "inputs", key_label: "label"}
        df = df.rename(columns=columns)  # type: ignore
        datasets_ds = datasets.Dataset.from_pandas(df)
        tok_ds = datasets_ds.map(self._tok_func, batched=True)
        if train:
            return datasets.DatasetDict(
                {
                    "train": tok_ds.select(indices_train),
                    "test": tok_ds.select(indices_validate),
                }
            )
        else:
            return tok_ds
