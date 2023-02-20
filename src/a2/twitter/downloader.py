""" code based on https://towardsdatascience.com/an-extensive-guide
-to-collecting-tweets-from-twitter-api-v2-for-academic-research
-using-python-3-518fcb71df2a"""
import logging
import os
import pathlib
import typing as t

import a2.twitter.twitter_api
import a2.utils.file_handling
import pandas as pd

FILE_LOC = pathlib.Path(__file__).parent
EMOJI_PATH = FILE_LOC / "../data/emoji/emoji_df.csv"
VOCABULARY_PATH = FILE_LOC / "../data/vocabularies/"


def get_emojis_from_vocab(vocabulary: t.Iterable, exclude: t.Optional[t.Iterable] = None) -> t.List:
    """
    Creates and returns list of emojis based on vocabulary (list of terms).

    Available emojis are taken from a csv file that lists
    all available emojis taken from
    https://www.kaggle.com/datasets/eliasdabbas/emoji-data-descriptions-codepoints
    Parameters:
    ----------
    vocabulary: list of terms (use get_vocabulary)
    exclude: terms excluded from vocab when creating emoji list

    Returns
    -------
    emoji list
    """
    if exclude is None:
        exclude = []
    vocabulary = [x for x in vocabulary if x not in exclude]
    df_emoji = pd.read_csv(EMOJI_PATH)
    # ?: to get matching group not a capturing group
    lis = df_emoji.emoji[df_emoji.name.str.contains(r"\b(?:" + "|".join(vocabulary) + r")\b")].to_list()
    return lis


def download_tweets(
    filepath: t.Union[str, pathlib.Path],
    keyword: str,
    start_dates: t.Union[t.Sequence, str],
    end_dates: t.Union[t.Sequence, str],
    fields: t.Optional[t.Sequence[str]] = None,
    max_results: int = 10,
    sleep_time: float = 1.0,
) -> None:
    """
    Download tweets and save specified fields to csv and
    dump all output to json file

    Need to have an environmental variable TOKEN with Twitter api token.
    Single requests only return 500 tweets such that multiple queries are
    required per search.
    Queries via twitters 'search all' endpoint (see
    https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
    for more details)
    Parameters:
    ----------
    filename: filename + ending for json and csv files.py
    keyword: keyword specifying query
    start_dates: single or list of start dates for query
                 in format YYYY-MM-DDTHH:mm:ssZ
    end_dates: single or list of end dates for query
               in format YYYY-MM-DDTHH:mm:ssZ
    fields: fields to convert from json to csv file to reduce filesize,
            if None default fields used
    foldername: foldername to save files.py
    max_results: number of maximum results
    sleep_time: time in sec waited after individual request,
                note a maximum of 500 tweets can be obtained per single request

    Returns
    -------
    dict of json response
    """
    if not isinstance(start_dates, list):
        start_dates = [start_dates]
    if not isinstance(end_dates, list):
        end_dates = [end_dates]
    if len(end_dates) != len(start_dates):
        raise ValueError(f"len(end_dates): {len(end_dates)} != len(start_dates): " f"{len(start_dates)}")
    headers = a2.twitter.twitter_api.create_headers_with_token()
    filepath = pathlib.Path(filepath)
    path_csv = filepath.with_suffix(".csv")
    path_json = filepath.with_suffix(".json")
    if os.path.isfile(path_json):
        logging.info(f"Appending json file: {path_json}!")
    else:
        logging.info(f"Creating json file: {path_json}!")
    a2.utils.file_handling.csv_create(
        path_csv,
        header=[
            f"# info for query for json file {os.path.split(path_json)[1]}\n",
            f"# keyword: {keyword}\n",
            "# dates:" f"{', '.join([x+'-->'+y for x,y in zip(start_dates, end_dates)])}" "\n",
        ],
        row=fields,
        check_is_file_or_empty=True,
    )
    a2.twitter.twitter_api._download_tweets(
        headers,
        path_json,
        keyword,
        start_dates,
        end_dates,
        max_results=max_results,
        sleep_time=sleep_time,
    )


def get_vocabulary(
    filepath: t.Union[str, pathlib.Path] = VOCABULARY_PATH / "weather_vocab_enchanted_precipitation.txt",
) -> list:
    """
    Loads vocabulary terms from file and returns them as a list.

    Idea is to base queries on terms collected in vocab and
    therefore easily reproduce data_manipulation sets.
    Parameters:
    ----------
    filepath: path to vocabulary file

    Returns
    -------
    list of vocab terms
    """
    with open(filepath) as f:
        vocabulary = [line.rstrip() for line in f]
    return vocabulary
