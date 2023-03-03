import os

import a2.training.dataset_hugging
import a2.training.evaluate_hugging
import a2.training.training_hugging
import a2.utils
import numpy as np
import sklearn.model_selection


def test_HuggingFaceTrainerClass_get_trainer(
    fake_dataset_training,
    tmp_path,
):
    # based on this test model: https://huggingface.co/stas/tiny-wmt19-en-de
    # as generated here:
    # https://huggingface.co/stas/tiny-wmt19-en-de/blob/main/fsmt-make-tiny-model.py

    folder_tracking = tmp_path / "mlflow/"
    folder_tracking.mkdir()
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
        folder_output=folder_output,
    )

    tracker = a2.training.tracking.Tracker()

    def get_raw_parameters():
        run = tracker.active_run()
        _tracking_uri = tracker.get_tracking_uri()
        print(f"{_tracking_uri=}")
        _run_artifact_dir = tracker.local_file_uri_to_path(_tracking_uri)
        folder_params = _run_artifact_dir + f"0/{run.info.run_id}/params/"
        parameters = {}
        for file in os.listdir(folder_params):
            with open(folder_params + file) as f:
                content = f.read()
                parameters[file] = a2.utils.utils.evaluate_string(content)  # if isinstance(content, str) else content
        return parameters

    tracker.set_tracking_uri("file://" + folder_tracking.__str__() + "/")
    experiment_id = tracker.create_experiment("experiment1")
    with tracker.start_run(experiment_id=experiment_id):
        tracker.log_params(trainer_object.hyper_parameters.__dict__)
        parameters = get_raw_parameters()
        assert parameters == trainer_object.hyper_parameters.__dict__
        trainer.train()
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
        ds_test = a2.training.evaluate_hugging.build_ds_test(
            ds=ds,
            indices_test=indices_validate,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
        )
        truth = ds_test.raining.values
        a2.training.tracking.log_metric_classification_report(tracker, truth, predictions, step=1)
        assert np.array_equal(predictions, np.array([1, 1, 1, 1]))
    (truth, predictions, prediction_probabilities,) = a2.training.evaluate_hugging.make_predictions_loaded_model(
        ds,
        indices_validate,
        folder_output + "checkpoint-1/",
        folder_tokenizer=folder_output + "checkpoint-1/",
        key_inputs="text_normalized",
        fp16=False,
    )
    assert np.array_equal(predictions, np.array([1, 1, 1, 1]))


def test_split_training_set(fake_dataset_training):
    indices_train, indices_validate = a2.training.training_hugging.split_training_set(
        fake_dataset_training,
        key_stratify="raining",
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    np.testing.assert_array_equal(indices_train, np.array([0, 7, 2, 4, 3, 6]))
    np.testing.assert_array_equal(indices_validate, np.array([1, 5]))
