import os
import pathlib
import stat
from contextlib import nullcontext as doesnotraise

import a2.utils.file_handling
import a2.utils.testing
import pytest
from pytest_cases import parametrize


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data/"


def test_get_header():
    header = a2.utils.file_handling.get_header(DATA_FOLDER / "test_locations_not_found.csv", n=1)
    assert header == ["# list of place_id's that were not found by twitter api\n"]


def test_json_load_error():
    with pytest.raises(ValueError):
        a2.utils.file_handling.json_load("fake_json.json")


def test_download_file(tmp_path):
    directory = tmp_path / "test_add_precipitation_memory_efficient/"
    directory.mkdir()

    a2.utils.file_handling.download_file(
        "https://cooking.stackexchange.com/questions/121426/can-you-shred-raw-potatoes-to-make-mashed-potatoes",  # noqa
        folder=directory.__str__(),
    )
    a2.utils.file_handling.download_file(
        "https://cooking.stackexchange.com/questions/121426/can-you-shred-raw-potatoes-to-make-mashed-potatoes",  # noqa
        folder=directory.__str__(),
    )  # test not overwrite
    a2.utils.file_handling.remove_existing_files(
        directory.__str__() + "utilscan-you-shred-raw-potatoes-to-make-mashed-potatoes"
    )


@parametrize(
    "chmod_flag, expected_rights",
    [
        (
            stat.S_IRUSR,
            ["cannot", "cannot", "can"],
        ),
        (
            stat.S_IWUSR | stat.S_IXUSR,
            ["can", "can", "cannot"],
        ),
    ],
)
def test_check_acess_rights_folder(chmod_flag, expected_rights, tmp_path):
    directory = tmp_path / "test_add_precipitation_memory_efficient/"
    directory.mkdir()
    expected_print = ""
    for right, task in zip(expected_rights, ["execute", "write", "read"]):
        expected_print += f"User {right} {task} files in folder {directory}\n"
    os.chmod(directory, chmod_flag)
    io_capture = a2.utils.testing.IOCapture()
    a2.utils.file_handling.check_acess_rights_folder(directory)
    printed = io_capture.return_capture_stop()
    assert printed == expected_print
    # make accessible to everyone so can be cleaned up by pytest
    os.chmod(directory, 0o777)


@parametrize(
    "folder_name, check_if_empty, raise_exception, expected",
    [
        (
            "fake_folder",
            False,
            True,
            ValueError(),
        ),
        (
            "tmp",
            True,
            True,
            ValueError(),
        ),
        (
            "tmp",
            False,
            False,
            True,
        ),
        (
            "tmp",
            True,
            False,
            False,
        ),
    ],
)
def test_folder_exists(folder_name, expected, tmp_path, check_if_empty, raise_exception):
    directory = tmp_path / "test_folder_exists/"
    directory.mkdir()
    if folder_name == "tmp":
        folder_name = directory
    with pytest.raises(type(expected)) if isinstance(expected, Exception) else doesnotraise():
        exists = a2.utils.file_handling.folder_exists(
            folder_name, check_if_empty=check_if_empty, raise_exception=raise_exception
        )
        assert exists == expected
