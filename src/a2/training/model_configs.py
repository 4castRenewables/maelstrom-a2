from typing import Literal

import a2.training.training_hugging
import a2.utils.utils
import transformers


SUPPORTED_MODELS: Literal = ["deberta_base", "deberta_small", "electra_base"]
SUPPORTED_TRAINERS: Literal = ["default", None]
SUPPORTED_LOSSES: Literal = ["default_loss", "focal_loss"]


def get_model_config(model_name: SUPPORTED_MODELS):
    if model_name == "deberta_base" or model_name == "deberta_small":
        hyper_parameters = a2.training.training_hugging.HyperParametersDebertaClassifier()
    elif model_name == "electra_base":
        hyper_parameters = a2.training.training_hugging.HyperParametersElectraClassifier()
    else:
        raise ValueError(f"{model_name=} not supported, ({SUPPORTED_MODELS=})!")
    return hyper_parameters


def get_customized_trainer_class(trainer_name: SUPPORTED_TRAINERS, method_overwrites: list | None = None):
    if method_overwrites is None:
        method_overwrites = []
    if trainer_name == "default" or trainer_name is None:
        trainer = transformers.Trainer
    else:
        raise ValueError(f"{trainer_name=} not supported ({SUPPORTED_MODELS=})!")

    if a2.utils.utils.is_in_list_and_remove("focal_loss", method_overwrites):
        trainer.compute_loss = a2.training.training_hugging.TrainerWithFocalLoss.compute_loss
    elif a2.utils.utils.is_in_list_and_remove("default_loss", method_overwrites):
        pass
    if len(method_overwrites) != 0:
        raise ValueError(f"{method_overwrites=} not supported ({SUPPORTED_LOSSES=})!")

    return trainer
