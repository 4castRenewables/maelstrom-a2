import numpy as np


def validate_array(x: np.ndarray, type: str = "float", name: str = None):
    """Validate numpy based on `type`, e.g. no nan-values"""
    if type == "float":
        if np.sum(np.isnan(x.astype(float))) > 0:
            raise ValueError(f"Found nan-values {f'in {name}' if name is not None else ''}!")


def assert_type(variable, type_to_check):
    if not isinstance(variable, type_to_check):
        raise TypeError(f"{variable} is of type: {type(variable)} != expected type: {type_to_check}")
