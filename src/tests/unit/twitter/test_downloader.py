""" based on https://opensource.com/article/21/9/unit-test-python"""
import pathlib
from contextlib import nullcontext as doesnotraise

import a2.dataset
import a2.twitter.downloader
import a2.utils.file_handling
import a2.utils.testing
import pytest
import responses
from pytest_cases import parametrize


FILE_LOC = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    "vocabulary, exclude, expected",
    [
        (["fog"], None, 2),
    ],
)
def test_get_emojis_from_vocab(vocabulary, exclude, expected):
    with pytest.raises(type(expected)) if isinstance(expected, Exception) else doesnotraise():
        lis = a2.twitter.downloader.get_emojis_from_vocab(vocabulary, exclude)
        assert len(lis) == expected


def test_get_vocabulary():
    assert isinstance(
        a2.twitter.downloader.get_vocabulary(
            FILE_LOC / "../../../a2/data/vocabularies/" "weather_vocab_enchanted_learning_com.txt"
        ),
        list,
    )
    assert isinstance(
        a2.twitter.downloader.get_vocabulary(
            FILE_LOC / "../../../a2/data/vocabularies/" "weather_vocab_enchanted_precipitation.txt"
        ),
        list,
    )


@parametrize(
    "fake_json_response, expected_dataframe, expected_csv_content",
    [
        ("fake_tweets_json_data", "fake_tweets_dataframe", "fake_csv_content_tweets_download"),
        ("fake_tweets_json_data_no_next_token", "fake_tweets_dataframe", "fake_csv_content_tweets_download"),
    ],
)
def test_mock_api_download_tweets(
    tmp_path,
    fake_json_response,
    expected_dataframe,
    expected_csv_content,
    request,
):
    directory = tmp_path / "test_download_tweets/"
    directory.mkdir()

    keyword = "sun has:geo -is:retweet (lang:en) " "place_country:GB -is:nullcast -from:3446146816"
    start_dates = "2020-01-1T10:00:00.000Z"
    end_dates = "2020-01-1T12:00:00.000Z"
    max_results = 2
    fields = None

    with responses.RequestsMock() as rsps:
        fake_json_response = request.getfixturevalue(fake_json_response)
        rsps.add(
            responses.GET,
            "https://api.twitter.com/2/tweets/search/all",
            json=fake_json_response[0],
            status=200,
        )
        filename = directory / "tweets"
        a2.twitter.downloader.download_tweets(
            filename,
            keyword,
            start_dates,
            end_dates,
            fields=fields,
            max_results=max_results,
            sleep_time=0.01,
        )
        expected_dataframe = request.getfixturevalue(expected_dataframe)
        expected_csv_content = request.getfixturevalue(expected_csv_content)

        df = a2.dataset.load_dataset.load_tweets_dataframe_from_jsons([f"{filename}.json"])
        a2.utils.testing.assert_equal_pandas_dataframe(df, expected_dataframe)
        assert a2.utils.file_handling.csv_open(directory / f"{filename}.csv") == expected_csv_content
