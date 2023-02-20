import logging
import math
import os
import pathlib
import time
import typing as t

import a2.utils.file_handling
import requests


def _get_token() -> str:
    """
    Returns twitter api token, which is saved as a global
    variable to authenticate user.
    """
    token = os.getenv("TOKEN")
    if token is None:
        raise ValueError("twitter api token: TOKEN not found as environment variable!")
    return token


def _create_headers(bearer_token: str) -> dict:
    """
    Creates header dict of url for the request based on bearer_token.
    """
    headers = {"Authorization": f"Bearer {bearer_token}"}
    return headers


def create_headers_with_token():
    """
    creates twitter api token and create header
    """
    bearer_token = _get_token()
    return _create_headers(bearer_token)


def _create_url_and_query_parameters(
    keyword: str,
    start_date: t.Optional[str] = None,
    end_date: t.Optional[str] = None,
    max_results: int = 10,
    search_url: str = "https://api.twitter.com/2/tweets/search/all",
) -> tuple[str, dict]:
    """
    Creates url for the request type search all.

    Parameters:
    ----------
    keyword: search keyword
    start_date: start date of tweets
    end_date: end date of tweets
    max_results: maximum number searched for

    Returns
    -------
    tuple(search url, query parameters)
    """

    query_params: dict = {
        "query": keyword,
        "start_time": start_date,
        "end_time": end_date,
        "max_results": max_results,
        "expansions": "author_id,in_reply_to_user_id,geo.place_id",
        "tweet.fields": "id,text,author_id,in_reply_to_user_id,geo,"
        "conversation_id,created_at,lang,public_metrics,"
        "referenced_tweets,reply_settings,source",
        "user.fields": "id,name,username,location,created_at," "description,public_metrics,verified",
        "place.fields": "full_name,id,country,country_code,geo,name," "place_type,contained_within",
        "next_token": None,
    }
    return search_url, query_params


def _connect_to_endpoint(
    url: str,
    headers: t.Mapping,
    params: dict,
    next_token: t.Optional[str] = None,
):
    """
    Connect to endpoint to request search query.

    Requests user query specified in url and header plus
    authentification from header.
    As single requests return a maximum of 500 tweets,
    need to save next token to keep quering for.
    Parameters:
    ----------
    url: url to query
    headers: headers of query
    params: query params (next token will be updated)
    next_token: will update former `next_token` in `params`

    Returns
    -------
    dict of json response
    """
    params["next_token"] = next_token  # params object received from create_url function
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def _download_tweets(
    headers: dict,
    path_json: t.Union[str, pathlib.Path],
    keyword: str,
    start_dates: t.Sequence[str],
    end_dates: t.Sequence[str],
    max_results: int = 10,
    sleep_time: float = 1.0,
) -> None:
    """
    Download tweets and save specified fields to csv and
    dump all output to json file

    Request tweets from twitter api and save full json response and
    save specified response fields.
    Need to have global variable TOKEN with twitter api token.
    Single requests only return 500 tweets such that multiple queries are
    required per search.
    Queries via twitters 'search all' endpoint
    (see https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all for more details) # noqa
    Parameters:
    ----------
    headers: header dict of url for the request based on bearer_token
    path_csv: path of .csv file
    path_json: path to .json file
    fields: fields to convert from json to csv file to reduce file size
    keyword: keyword specifying query
    start_dates: list of start dates for query in format YYYY-MM-DDTHH:mm:ssZ
    end_dates: list of end dates for query in format YYYY-MM-DDTHH:mm:ssZ
    max_results: number of maximum results
    sleep_time: time in sec waited after individual request, note a maximum of 500 tweets can be obtained per single request

    Returns
    -------

    """
    max_count = math.ceil(max_results / len(start_dates))  # Max tweets per time period
    total_tweets = 0
    for i_dates, (start_date, end_data) in enumerate(zip(start_dates, end_dates)):
        count = 0  # Counting tweets per time period
        stay_this_time = True
        next_token = None
        max_count_request = min([max_count, 500])
        while stay_this_time:
            # Check if max_count reached
            if count >= max_count:
                break
            if count + max_count_request > max_count:
                max_count_request = max([int(max_count - count), 10])
            url, params = _create_url_and_query_parameters(keyword, start_date, end_data, max_count_request)
            json_response = _connect_to_endpoint(url, headers, params, next_token)
            result_count = json_response["meta"]["result_count"]

            _save_response_to_json(json_response, path_json)
            if "next_token" in json_response["meta"]:
                # Save the token to use for next call
                next_token = json_response["meta"]["next_token"]
                if result_count is not None and result_count > 0 and next_token is not None:
                    count, total_tweets = _increment_counters(count, result_count, total_tweets)
                    logging.info(f"Total # of Tweets added: {total_tweets}")
                    time.sleep(sleep_time)
                    # If no next token exists
            else:
                if result_count is not None and result_count > 0:
                    count, total_tweets = _increment_counters(count, result_count, total_tweets)
                    time.sleep(sleep_time)

                # Since this is the final request, turn flag to false
                # to move to the next time period.
                stay_this_time = False
                next_token = None
            time.sleep(sleep_time)
    logging.info(f"Total number of results: {total_tweets}")


def _increment_counters(count: int, result_count: int, total_tweets: int) -> t.Tuple[int, int]:
    count += result_count
    total_tweets += result_count
    return count, total_tweets


def _save_response_to_json(json_response: dict, path_json: t.Union[str, pathlib.Path]):
    a2.utils.file_handling.json_dump(path_json, list(), check_is_file=True)
    data = a2.utils.file_handling.json_load(path_json)
    data.append(json_response)
    a2.utils.file_handling.json_dump(path_json, data)
