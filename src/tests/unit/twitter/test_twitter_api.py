import a2.twitter.twitter_api
import pytest
import requests


def test__auth(monkeypatch):
    monkeypatch.delenv("TOKEN")
    with pytest.raises(ValueError):
        a2.twitter.twitter_api._get_token()


def test__connect_to_endpoint():
    with pytest.raises(type(requests.exceptions.HTTPError())):
        headers = a2.twitter.twitter_api.create_headers_with_token()
        url = a2.twitter.twitter_api._create_url_and_query_parameters("test fake language (lang:fake)")
        a2.twitter.twitter_api._connect_to_endpoint(url[0], headers, url[1], next_token=None)
