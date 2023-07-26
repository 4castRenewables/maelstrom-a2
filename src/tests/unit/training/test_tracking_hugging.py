import os

import a2.training.dataset_hugging
import a2.training.evaluate_hugging
import a2.training.tracking
import a2.training.tracking_hugging
import a2.training.training_hugging
import a2.utils
import numpy as np
import sklearn.model_selection


def test_LogCallback(
    fake_dataset_training,
    capsys,
    tmp_path,
):
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
    folder_output = tmp_path / "model_output/"
    folder_output.mkdir()
    folder_output = folder_output.__str__() + "/"

    model_folder = "google/bert_uncased_L-2_H-128_A-2"
    ds = fake_dataset_training
    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(model_folder)
    indices_train, indices_validate = sklearn.model_selection.train_test_split(
        np.arange(ds["index"].shape[0]),
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=ds.raining.values,
    )
    dataset = dataset_object.build(
        ds,
        indices_train=indices_train,
        indices_validate=indices_validate,
        key_inputs="text_normalized",
    )
    trainer_object = a2.training.training_hugging.HuggingFaceTrainerClass(model_folder, config=None)
    trainer = trainer_object.get_trainer(
        dataset,
        fp16=False,
        mantik=False,
        evaluation_strategy="epoch",
        folder_output=folder_output,
        hyper_parameters=a2.training.training_hugging.HyperParametersDebertaClassifier(epochs=2),
        callbacks=[a2.training.tracking_hugging.LogCallback(tracker=None)],
    )
    trainer.train()
    captured = capsys.readouterr().out
    test_ds = dataset_object.build(
        ds,
        indices_train,
        indices_validate,
        train=False,
        key_inputs="text_normalized",
    )
    (
        predictions,
        prediction_probabilities,
    ) = a2.training.evaluate_hugging.predict_dataset(test_ds, trainer)
    assert len(captured) > 720 and len(captured) < 770
    assert np.array_equal(predictions, np.array([1, 1, 1, 1]))
