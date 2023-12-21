from typing import Literal
from typing import Optional

import a2.training.training_hugging
import a2.utils.utils
import transformers


SUPPORTED_MODELS: Literal = ["deberta_base", "deberta_small", "electra_base"]
SUPPORTED_TRAINERS: Literal = ["default", "deep500", None]
SUPPORTED_LOSSES: Literal = ["default_loss", "focal_loss"]


def update_hyperparameters():
    return


def get_model_config(model_name: SUPPORTED_MODELS, parameters_overwrite: Optional[dict] = None):
    """
    Get data class that holds hyper parameters of models that are specified via their name (str)

    Parameters:
    ----------
    model_name: Name of model, only selection available, see `a2.training.model_configs.SUPPORTED_MODELS`
    parameters_overwrite: Overwrites hyper parameters based on dictionary key-value pairs,
        ignores keys not present in original data class

    Returns
    -------
    dictionary of metrics
    """
    if parameters_overwrite is None:
        parameters_overwrite = {}
    if model_name == "deberta_base" or model_name == "deberta_small":
        hyper_parameters = a2.training.training_hugging.HyperParametersDebertaClassifier()
    elif model_name == "electra_base":
        hyper_parameters = a2.training.training_hugging.HyperParametersElectraClassifier()
    else:
        raise ValueError(f"{model_name=} not supported, ({SUPPORTED_MODELS=})!")
    hyper_parameters.update(parameters_overwrite)
    return hyper_parameters


def get_customized_trainer_class(trainer_name: SUPPORTED_TRAINERS, method_overwrites: list | None = None):
    if method_overwrites is None:
        method_overwrites = []
    if trainer_name == "default" or trainer_name is None:
        trainer = transformers.Trainer
    if trainer_name == "deep500" or trainer_name is None:
        import a2.training.training_deep500

        trainer = a2.training.training_deep500.TrainerWithTimer
    else:
        raise ValueError(f"{trainer_name=} not supported ({SUPPORTED_MODELS=})!")

    if a2.utils.utils.is_in_list_and_remove("focal_loss", method_overwrites):
        trainer.compute_loss = a2.training.training_hugging.TrainerWithFocalLoss.compute_loss
    elif a2.utils.utils.is_in_list_and_remove("default_loss", method_overwrites):
        pass
    if len(method_overwrites) != 0:
        raise ValueError(f"{method_overwrites=} not supported ({SUPPORTED_LOSSES=})!")

    return trainer
