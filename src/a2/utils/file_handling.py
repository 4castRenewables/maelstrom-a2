import collections.abc
import csv
import fnmatch
import glob
import json
import logging
import os
import pathlib
import tarfile
import typing as t
import zipfile

import requests
import tqdm


def csv_append(filename: t.Union[str, pathlib.Path], row: list) -> None:
    """
    append row to csv file

    Parameters:
    ----------
    filename: path to file
    row: row to add to file

    Returns
    -------
    """
    with open(filename, "a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row)


def json_dump(
    filename: t.Union[str, pathlib.Path],
    data: object,
    check_is_file: bool = False,
    log_if_new_file: bool = False,
) -> None:
    """
    dump data to json file

    Parameters:
    ----------
    filename: path to file
    data: data to dump
    check_is_file: if True, only writes file if it doesn't exist
    log_if_new_file: if True, logs creation of new file

    Returns
    -------
    """
    if not check_is_file or (check_is_file and not os.path.isfile(filename)):
        if not os.path.isfile(filename) and log_if_new_file:
            logging.info(f"initializing new file: {filename}")
        with open(filename, mode="w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def json_load(filename: t.Union[str, pathlib.Path], raise_if_none: bool = True) -> list:
    """
    load json file

    Parameters:
    ----------
    filename: path to file

    Returns
    -------
    """
    if raise_if_none and not os.path.isfile(filename):
        raise ValueError(f"can't load file: {filename} as doesn't exist!")
    return json.loads(open(filename).read())


def csv_create(
    filename: t.Union[str, pathlib.Path],
    header: t.Iterable[str] | None = None,
    row: t.Iterable[str] | None = None,
    check_is_file_or_empty: bool = True,
) -> None:
    """
    initializes csv file, adds header, row (optional)

    Parameters:
    ----------
    filename: path to file
    header: header as list of strings
    row: row to add to file

    Returns
    -------
    """
    if not check_is_file_or_empty or (
        check_is_file_or_empty and (not os.path.isfile(filename) or not os.path.getsize(filename))
    ):
        logging.info(f"creating new csv file: {filename}")
        with open(filename, "w", newline="", encoding="utf-8") as fp:
            if header is not None:
                for h in header:
                    fp.write(h)
            csv_writer = csv.writer(fp, escapechar="\\")
            if row is not None:
                csv_writer.writerow(row)


def get_header(filepath: t.Union[str, pathlib.Path], n: int = 2) -> list[str]:
    """
    return header of file where header is n lines long

    Parameters:
    ----------
    filepath: path to file
    n: number of rows covering header

    Returns
    -------
    header of file
    """
    header = []
    with open(filepath) as f:
        for i in range(n):
            header.append(f.readline())
    return header


def csv_open(filename: t.Union[str, pathlib.Path]) -> list:
    """
    append row to csv file

    Parameters:
    ----------
    filename: path to file
    row: row to add to file

    Returns
    -------
    """
    with open(filename) as read_obj:
        csv_reader = csv.reader(read_obj)
        return [row[0] if len(row) == 1 else row for row in csv_reader]


def download_file(url: str, folder: str | None = None, overwrite: bool = False) -> str:
    """
    Download file based on url to folder

    File with same name can be overwritten by setting `overwrite` to True
    Parameters:
    ----------
    url: url of file
    folder: folder where file saved
    overwrite: overwrite file with same name by downloaded file

    Returns
    -------
    """
    local_filename = ""
    if folder is not None:
        local_filename = folder
    local_filename = local_filename + url.split("/")[-1]
    if os.path.isfile(local_filename) and not overwrite:
        logging.info(f"file: {local_filename} already exists, no download required.")
        return local_filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
    return local_filename


def unzip_file(
    path_to_zip_file: t.Union[str, pathlib.Path],
    directory_to_extract_to: t.Union[str, pathlib.Path],
) -> None:
    """Unzip file to directory"""
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def decompress_tarball(filename: t.Union[str, pathlib.Path], to_folder: t.Union[str, pathlib.Path]):
    """Decompress tarball to folder"""
    with tarfile.open(filename) as tar:
        tar.extractall(path=to_folder)


def get_all_files(
    path: t.Union[str, pathlib.Path],
    include_subdirectories: bool = False,
    recursive: bool = False,
):
    """
    Find files based on path pattern.

    Parameters:
    ----------
    path: Base path pattern, which will be used to search directory
    include_subdirectories: Include sub directories in search
    recursive: Make a recursive search (if include_subdirectories=False)

    Returns
    -------
    List of filenames
    """
    path = pathlib.Path(path)
    if include_subdirectories:
        all_files = []
        root = path.parent
        pattern = path.name
        for current_path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    all_files.append(os.path.join(current_path, name))
    else:
        path = str(path)
        all_files = glob.glob(path, recursive=recursive)
    return all_files


def remove_existing_files(files_wildcards: t.Union[str, list[str]], include_subdirectories: bool = False) -> None:
    """Remove files based on wildcards. Optionally include subdirectories in search for pattern."""
    if not isinstance(files_wildcards, collections.abc.Iterable) or isinstance(files_wildcards, str):
        files_wildcards = [files_wildcards]
    for fw in files_wildcards:
        files = get_all_files(fw, include_subdirectories=include_subdirectories)
        for f in files:
            if os.path.isfile(f):
                os.remove(f)


def file_exists_and_not_empty(filename: t.Union[str, pathlib.Path]) -> bool:
    """Check if file exists and is not empty"""
    return os.path.exists(filename) and not os.path.getsize(filename)


def sort_filenames_by_pattern(file_list: list, pattern_extraction_function: t.Callable) -> list:
    return sorted(
        file_list,
        key=lambda x: pattern_extraction_function(os.path.split(x)[1]),
    )


def is_jsc():
    """Check if currently working on the JSC supercomputer"""
    return os.path.isdir("/p/scratch/deepacf")


def get_folder_data():
    """Get the name of the folder where data for this project is stored indepent of machine used"""
    if is_jsc():
        folder_data = pathlib.Path("/p/scratch/deepacf/maelstrom/maelstrom_data/ap2/data/")
    else:
        folder_data = pathlib.Path("/home/kristian/Projects/a2/data/")
    if not os.path.isdir(folder_data):
        raise ValueError(f"{folder_data=} does not exist!")
    return folder_data


def get_folder_models():
    """Get the name of the folder where models for this project are stored indepent of machine used"""
    if is_jsc():
        folder_models = pathlib.Path("/p/scratch/deepacf/maelstrom/maelstrom_data/ap2/models_output/")
    else:
        folder_models = pathlib.Path("/home/kristian/Projects/a2/models/")
    if not os.path.isdir(folder_models):
        raise ValueError(f"{folder_models=} does not exist!")
    return folder_models


def get_folder_radar(folder_radar: t.Optional[pathlib.Path] = None) -> pathlib.Path:
    """Get folder where radar data is saved"""
    if folder_radar is None:
        folder_radar = get_folder_data() / "precipitation/radar/"
    return folder_radar


def check_acess_rights_folder(file_or_folder):
    if os.path.isdir(file_or_folder):
        description = "files in folder"
    else:
        description = "file"
    if os.access(file_or_folder, os.X_OK):
        print(f"User can execute {description} {file_or_folder}")
    else:
        print(f"User cannot execute {description} {file_or_folder}")
    if os.access(file_or_folder, os.W_OK):
        print(f"User can write {description} {file_or_folder}")
    else:
        print(f"User cannot write {description} {file_or_folder}")
    if os.access(file_or_folder, os.R_OK):
        print(f"User can read {description} {file_or_folder}")
    else:
        print(f"User cannot read {description} {file_or_folder}")


def check_folder_exists(path, check_is_empty=False, raise_exception=False):
    if not os.path.isdir(path):
        if raise_exception:
            raise ValueError(f"{path=} doesn't exist!")
        return False
    if check_is_empty and not os.path.getsize(path):
        if raise_exception:
            raise ValueError(f"{path=} is empty!")
        return False
    return True
