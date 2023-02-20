import logging
import pathlib
import re
import typing as t

import a2.dataset.load_dataset
import a2.utils
import numpy as np
import pandas as pd
import xarray

FILE_LOC = pathlib.Path(__file__).parent
EMOJI_PATH = FILE_LOC / "../data/emoji/emoji_df.csv"

COMPLEX_PUNCTUTATIONS = "@#+&*[]-%:/();$=><|{}^•…`*'!~"
COMPLEX_PUNCTUTATIONS += '"'
PUNCTUATIONS = COMPLEX_PUNCTUTATIONS + "!?.,"


def remove_non_ascii_characters(text: str) -> str:
    """
    limit your characters to ASCII characters

    ASCII (American Standard Code for Information Interchange)
    is the most common character encoding format for text
    data in computers and on the internet. In standard ASCII-encoded data,
    there are unique values for 128 alphabetic,
    numeric or special additional characters and control codes.
    """
    return text.encode("ascii", errors="ignore").decode()


def remove_unwanted_unicode(text: str) -> str:
    """
    remove emojis, symbols, flags and further unicode characters

    redundant when using `remove_non_ascii_characters`

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002460-\U000024FF"  # alpha numerics
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", text)


def remove_punctutations_func(text: str, method: str = "keep_basic_punctuations") -> str:
    """
    Removes punctuation characters with different methods

    Parameters:
    ----------
    text: text
    method: 'keep_basic_punctuations' (retain '!?.,'), 'all' (remove all)

    Returns
    -------
    cleaned text
    """
    if method == "keep_basic_punctuations":
        return re.sub(f"[{r''.join(re.escape(COMPLEX_PUNCTUTATIONS))}]", "", text)
    elif method == "all":
        return re.sub(f"[{r''.join(re.escape(PUNCTUATIONS))}]", "", text)
    else:
        raise Exception(f"method: {method} not understood")


def normalize_slang_stopwords(text: str, replace_underscore: bool = True) -> str:
    """
    Removes punctuation characters with different methods

    Parameters:
    ----------
    text: text
    replace_underscore: replace any number of
                        underscore occurences with a single space

    Returns
    -------
    cleaned text
    """
    text = re.sub("\u2014", "_", text)
    if replace_underscore:
        text = re.sub("_+", " ", text)
    text = re.sub(r"&amp;", " and ", text)
    # html way of using `&` symbol
    # see https://stackoverflow.com/questions/9084237/what-is-amp-used-for
    text = re.sub(r"’", "'", text)
    text = re.sub(r"“", "'", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text = re.sub("([Aa])ren't", "\\1re not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"here's", "here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"photogroup", "photo group", text)
    text = re.sub(r"youve", "you have", text)
    text = re.sub(r"donå«t", "do not", text)
    text = re.sub(r"\s\s+", " ", text)
    text = re.sub("\u2026", "...", text)
    text = re.sub("\u2060", "...", text)
    text = re.sub("Unιted", "United", text)
    text = re.sub(r"\S+\d+\S+|\S+\d+|\d+|\d+\S+", "", text)
    text = text.replace("...", " ... ")
    if "..." not in text:
        text = text.replace("..", " ... ")
    text = re.sub("\u2063", "", text)
    return text


def remove_urls(text: str) -> str:
    """
    Remove URL

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    text = re.sub(r"\S*https?:\S*", "", text, flags=re.MULTILINE)
    return text


def split_punctuation_text(text: str) -> str:
    """
    Introduce space between punctutation characters: .,!?;

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    text = re.sub(r"([\w']|)([\.,!\?;])", r"\1 \2", text)
    return text


def lower_case(text: str) -> str:
    """
    Convert text to lower case

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    return text.lower()


def remove_initial_final_space(text: str) -> str:
    """
    Remove space at beginning and/or end of sentence if available

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    text = re.sub("^ ", "", text)
    text = re.sub(" $", "", text)
    return text


def remove_hashtags(text: str) -> str:
    """
    Remove hashtags and mentions of other Twitter users (prefix "@")

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    return re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+", "", text, flags=re.MULTILINE)


def remove_instagram_atsign(text: str) -> str:
    """
    Remove suffix of Tweets posted via Instagram

    When users link their Twitter and Instagram account and post on Instagram,
    the Tweet will be appended with an at sign followed by tags and URLs,
    which may be divided by single spaces.
    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    return re.sub(
        r"@\s[\SA-Za-z0-9\s]+$",
        "",
        text,
        flags=re.MULTILINE,
    )


def reduce_punctations_to_single_occurence(text: str, pattern: re.Pattern) -> str:
    """
    Remove multiple occurences of the same punctuation
    symbol with a single space

    Parameters:
    ----------
    text: text
    pattern: pattern used to match multiple occurence of punctuations above 1

    Returns
    -------
    cleaned text
    """

    def cleanup_results(x):
        group = x.group()
        group_no_whitespace = group.replace(" ", "")
        return group_no_whitespace[0] + " " if group_no_whitespace else " "

    return pattern.sub(lambda x: cleanup_results(x), text)


def remove_instagram_abbreviation(text: str) -> str:
    """
    Remove suffix of Tweets when abbreviated for Tweets posted via Instagram

    When users link their Twitter and Instagram account and post on Instagram,
    the Tweet may be abbreviated where the remainder of the message is replaced
    by the special character "…", which may be followed by an at sign followed
    by the Instagram URL.
    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    return re.sub(r"@?… https:\S*$", "", text)


def replace_emojis(text: str, emojis_dic: t.Mapping, pattern: re.Pattern) -> str:
    """
    Replace emojis with their name as given in EMOJI_PATH

    Parameters:
    ----------
    text: text
    emojis_dic: dictionary mapping emoji symbol to name
    pattern: regex pattern to match emoji symbols

    Returns
    -------
    text where emojis replaced with their name
    """
    return pattern.sub(lambda x: emojis_dic[x.group().lower()], text)


def remove_whitespaces(text: str) -> str:
    """
    Replace more than single occurences of whitespaces with with single space

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    try:
        text = re.sub(r"\s\s+", " ", text)
    except Exception as e:
        raise Exception(f"Cannot remove whitespace from {text}") from e
    return text


def remove_newlines(text: str) -> str:
    """
    Replace new lines with single space

    Parameters:
    ----------
    text: text

    Returns
    -------
    cleaned text
    """
    text = re.sub("/\r?\n|\r/", "", text)
    text = re.sub("\n", " ", text)
    return text


def replace_hashtags_keywords(text: str, keywords_hashtag_dict: t.Mapping, pattern: re.Pattern) -> str:
    """
    Replace hashtag that match keywords exactly (e.g., #rain, #sun)
    with the respective keyword (e.g., rain, sun)

    Parameters:
    ----------
    text: text
    keywords_hashtag_dict: dictionary mapping hashtag keyword to keyword
    pattern: matching keyword including hashtag

    Returns
    -------
    cleaned text
    """
    return pattern.sub(lambda x: keywords_hashtag_dict[x.group().lower()], text)


def get_emojis_and_dictionary(scope: str = "all", keywords: t.Optional[t.Sequence] = None) -> t.Tuple[str, t.Mapping]:
    """
    Returns emojis and a dictionary with their symbol (keys) and name (values)

    Emojis are loaded from the csv file in `EMOJI_PATH`.
    Parameters:
    ----------
    scope: "all" (returns all emojis), "keywords"
    (returns all emojis whose name contains any keyword)
    keywords: list of keywords (used when scope='keywords')

    Returns
    -------
    string containing all emojis, dictionary mapping
    between emoji symbol and emoji name
    """
    df_emoji = _get_emojis_dataframe(filename=EMOJI_PATH)
    if scope == "all":
        emojis = "".join(df_emoji.emoji.values)
        emojis_dic = {
            k: v for k, v in zip(df_emoji.emoji.values, df_emoji.name.values) if k not in ["*️⃣", "*⃣"]
        }  # latter cause error for re.sub
    elif scope == "keywords":
        if keywords is None:
            raise ValueError(f"Need to specify keywords: {keywords} to filter emoji dic!")
        df_emoji_include_keywords = df_emoji.loc[df_emoji.name.str.contains("|".join(rf"\b{x}\b" for x in keywords))]
        emojis_dic = {
            x: f" {y} "
            for x, y in zip(
                df_emoji_include_keywords["emoji"].values,
                df_emoji_include_keywords["name"].values,
            )
        }
        emojis = "".join(df_emoji_include_keywords.emoji.values)
    else:
        raise ValueError(f"Specified scope: {scope} unknown!")
    return emojis, emojis_dic


def _get_emojis_dataframe(filename: t.Union[str, pathlib.Path] = EMOJI_PATH) -> pd.DataFrame:
    """
    Loads emoji dataframe from csv file (by default stored in `EMOJI_PATH`)

    Parameters:
    ----------
    filename: filename of csv file containing emoji dataframe

    Returns
    -------
    dataframe with emojis symbols, emoji names, etc.
    """
    df_emoji = pd.read_csv(filename)
    return df_emoji


class Normalizer:
    """
    Class used to normalize Sentences (like Tweets)
    """

    def __init__(self, keywords: t.Optional[t.Sequence] = None):
        """
        Initializes class including storing kewyords, loading emoji dataframe
        and compiling some regex

        Parameters:
        ----------
        keywords: list of keywords, used to replace emojis that contain
        keyword (by default)

        Returns
        -------
        """
        if keywords is None:
            keywords = a2.dataset.utils_dataset.get_keywords_default()
        self.keywords = keywords
        self.keywords_hashtag_dict = {f"#{k}": k for k in keywords}
        self.emojis, self.emojis_dic = get_emojis_and_dictionary(scope="keywords", keywords=self.keywords)
        # taken from
        # https://stackoverflow.com/questions/51102201/replace-a-string-using-dictionary-regex
        self.emojis_replace_pattern = re.compile(
            r"(" + r"|".join(re.escape(key) for key in self.emojis_dic.keys()) + r")",
            flags=re.UNICODE,
        )
        self.keywords_replace_pattern = re.compile(
            r"(" + r"|".join(re.escape(key) for key in self.keywords_hashtag_dict.keys()) + r")",
            flags=re.IGNORECASE,
        )
        self.stopword_single_occurence_pattern = re.compile(
            r"([\s+" + r"".join(re.escape(key) for key in PUNCTUATIONS) + r"]" + r"{2,})",
            flags=re.UNICODE,
        )

    def normalize(
        self,
        sentence: str,
        source: t.Optional[str] = None,
        ignore_non_ascii: bool = True,
        replace_keyword_emojis: bool = True,
        remove_punctuations: str = "keep_basic_punctuations",
        reduce_punctuations: bool = True,
        use_lower_case: bool = False,
        do_split_punctuation_text: bool = False,
    ) -> str:
        """
        Normalizes sentence.

        Order of operations matters! Keep this in mind when implementing
        new functionalities or introducing a change.
        Parameters:
        ----------
        sentence: sentence (Tweet)
        source: source of the Tweet. Note, special formatting required for Tweet
                suffix when `source='Instagram'`
        ignore_non_ascii: Remove non-ascii format characters that remain after
                          initial normalization of emojis and unusual usage of
                          some punctuations
        replace_keyword_emojis: Replaces emojis whose name contains keywords
                                with that keyword
        remove_punctuations: 'keep_basic_punctuations' or 'all'
        reduce_punctuations: Convert multiple occurence of same punctuation
                             character to single character.
        use_lower_case: Convert text to lower case
        do_split_punctuation_text: Split punctutation from text, which is
                                   required by some embeddings

        Returns
        normalized sentence
        -------
        """
        try:
            if source is not None and source == "Instagram":
                sentence = remove_instagram_atsign(sentence)
                sentence = remove_instagram_abbreviation(sentence)
            if replace_keyword_emojis:
                sentence = replace_emojis(
                    sentence,
                    self.emojis_dic,
                    pattern=self.emojis_replace_pattern,
                )
            sentence = replace_hashtags_keywords(
                sentence,
                self.keywords_hashtag_dict,
                pattern=self.keywords_replace_pattern,
            )
            sentence = remove_hashtags(sentence)
            sentence = normalize_slang_stopwords(sentence, replace_underscore=True)
            sentence = remove_unwanted_unicode(sentence)
            sentence = remove_newlines(sentence)
            sentence = remove_urls(sentence)
            if remove_punctuations:
                sentence = remove_punctutations_func(sentence, method=remove_punctuations)
            if ignore_non_ascii:
                sentence = remove_non_ascii_characters(sentence)
            if reduce_punctuations:
                sentence = reduce_punctations_to_single_occurence(
                    sentence, pattern=self.stopword_single_occurence_pattern
                )
            if use_lower_case:
                sentence = lower_case(sentence)
            sentence = remove_whitespaces(sentence)
            sentence = remove_initial_final_space(sentence)
            if do_split_punctuation_text:
                sentence = split_punctuation_text(sentence)
        except Exception as e:
            raise Exception(f"Cannot process: {sentence}") from e
        return sentence


def normalize_text_dataset(
    ds: xarray.Dataset,
    keywords: t.Optional[t.Sequence] = None,
    key_text_original: str = "text_original",
    key_text_normalized: str = "text_normalized",
    key_text_backup: t.Optional[str] = None,
    ignore_non_ascii: bool = True,
    replace_keyword_emojis: bool = True,
    remove_punctuations: str = "keep_basic_punctuations",
    reduce_punctuations: bool = True,
    use_lower_case: bool = False,
    do_split_punctuation_text: bool = False,
    processes: int = -1,
) -> xarray.Dataset:
    """
    Normalizes all sentences in dataset and adds normalized sentences to dataset

    Order of operations matters! Keep this in mind when implementing new
    functionalities or introducing a change. Transforming text in field
    'key_text_original', backing it up to field 'key_text_backup' (if specified)
    and saving it to 'key_text_normalized'
    (if specified, otherwise 'key_text_original' is overwritten)
    Parameters:
    ----------
    ds: xarray dataset
    keywords: List of keywords, used to replace emojis that contain
              keyword (by default)
    key_text_original: Variable name in dataset for text to be
                       normalized
    key_text_normalized: Variable name in dataset for normalized
                         text to be saved as
    key_text_backup: Variable name the raw text will be stored
                     in as a backup
    The following keyword arguments correspond to keyord arguments
    of Normalizer.normalize method:
        ignore_non_ascii: Remove non-ascii format characters that
                          remain after initial normalization of
                          emojis and unusual usage of some punctuations
        replace_keyword_emojis: Replaces emojis whose name contains
                                keywords with that keyword
        remove_punctuations: 'keep_basic_punctuations' or 'all'
        reduce_punctuations: Convert multiple occurence of same
                             punctuation character to single character.
        use_lower_case: Convert text to lower case
        do_split_punctuation_text: Split punctutation from text, which
                                   is required by some embeddings
    processes: Number of processes used to normalize text

    Returns
    dataset with normalized text and original text
    -------
    """
    ds = a2.dataset.load_dataset.reset_index_coordinate(ds)
    if keywords is None:
        keywords = a2.dataset.utils_dataset.get_keywords_default()
    normalizer = Normalizer(keywords=keywords)
    logging.info(f"emojis_dic: {normalizer.emojis_dic}")
    logging.info(f"keywords: {keywords}")
    kwargs = {
        "ignore_non_ascii": ignore_non_ascii,
        "replace_keyword_emojis": replace_keyword_emojis,
        "remove_punctuations": remove_punctuations,
        "reduce_punctuations": reduce_punctuations,
        "use_lower_case": use_lower_case,
        "do_split_punctuation_text": do_split_punctuation_text,
    }
    normalized_text = a2.utils.utils.parallelize(
        function=normalizer.normalize,
        args_zipped=zip(ds[key_text_original].values, ds.source.values),
        kwargs_as_dict=dict(**kwargs),
        processes=processes,
    )
    if key_text_backup is not None:
        ds[key_text_backup] = (["index"], ds[key_text_original].values.copy())
    index_max = ds.index.shape[0]
    indices = np.linspace(0, index_max, 24, dtype=int)

    if key_text_normalized is not None and key_text_normalized not in ds.variables.keys():
        ds[key_text_normalized] = (
            ["index"],
            np.full_like(ds["index"].values, "", dtype=object),
        )
    if key_text_normalized is None:
        key_text_normalized = key_text_original

    for index_start, index_end in zip(indices[:-1], indices[1:]):
        mask = (ds.index >= index_start) & (ds.index <= index_end)
        ds[key_text_normalized].loc[mask] = normalized_text[slice(index_start, index_end + 1)]
    return ds


def filter_text_dataset(
    ds: xarray.Dataset,
    key_text: str = "text_normalized",
    remove_sun_confusing_terms: bool = True,
    only_text_containing_keywords: bool = True,
    maximum_bounding_box_area: t.Optional[float] = 100,
    only_unique_text: bool = True,
    keywords: t.Sequence = None,
) -> xarray.Dataset:
    """
    Filters Tweets to allow for more concise (problem specific) text
    in training dataset

    Parameters:
    ----------
    ds: xarray dataset
    remove_sun_confusing_terms: Remove Tweets containing "Sun" as it
                                is commonly used to refer to the British
                                newspaper "The Sun" or Sunday
    only_text_containing_keywords: Remove Tweets containing no element of
                                   `keywords`
    keywords: List of keywords, if None
              using `a2.dataset.utils_dataset.get_keywords_default`
    maximum_bounding_box_area: Removes Tweets with
                               bounding_box_area > maximum_bounding_box_area
                               if `None` ignored
    only_unique_text: Removes Tweets with duplicate text

    Returns
    filtered dataset
    -------
    """
    if remove_sun_confusing_terms:
        ds = ds.where(~ds[key_text].str.contains(r"(?<!^)\bSun\b"), drop=True)
    if only_text_containing_keywords:
        if keywords is None:
            keywords = a2.dataset.utils_dataset.get_keywords_default()
        ds = ds.where(
            ds[key_text].str.contains("|".join(keywords), flags=re.IGNORECASE),
            drop=True,
        )
    if maximum_bounding_box_area is not None and "bounding_box_area" in ds.variables.keys():
        ds = ds.where(
            ds["bounding_box_area"]
            < maximum_bounding_box_area | a2.dataset.utils_dataset.is_nan(ds, "bounding_box_area"),
            drop=True,
        )
    if only_unique_text:
        ds = a2.dataset.load_dataset.reset_index_coordinate(ds)
        _, indices, indices_inverse = np.unique(ds[key_text].values, return_index=True, return_inverse=True)
        ds = ds.sel(index=indices)
    ds = a2.dataset.load_dataset.reset_index_coordinate(ds)
    return ds


def normalize_filter_dataset(
    ds: xarray.Dataset,
    keywords: t.Optional[t.Sequence] = None,
    reset_index: bool = True,
    key_text_original: str = "text_original",
    key_text_normalized: str = "text_normalized",
    key_text_backup: t.Optional[str] = None,
    ignore_non_ascii: bool = True,
    replace_keyword_emojis: bool = True,
    remove_punctuations: str = "keep_basic_punctuations",
    reduce_punctuations: bool = True,
    use_lower_case: bool = False,
    do_split_punctuation_text: bool = True,
    remove_sun_confusing_terms: bool = True,
    only_text_containing_keywords: bool = True,
    maximum_bounding_box_area: t.Optional[float] = 100,
    only_unique_text: bool = True,
    processes: int = -1,
) -> xarray.Dataset:
    """
    Normalizes all sentences in dataset and filter out unwanted sentences
    from dataset

    Shorthand function combining `normalize_text_dataset` and
    `filter_text_dataset`. Normalizes text and filters based on
    normalized text.
    Parameters:
    ----------
    ds: xarray dataset
    keywords: List of keywords, used to replace emojis that
              contain keyword (by default), if `None` using
              `a2.dataset.utils_dataset.get_keywords_default`
    reset_index: Resetting index coordinate of final dataset
    Normalization:
    key_text_original: Variable name in dataset for text to
                       be normalized
    key_text_normalized: Variable name in dataset for normalized
                         text to be saved as
    key_text_backup: Variable name the raw text will be stored
                     in as a backup
    ignore_non_ascii: Remove non-ascii format characters that
                      remain after initial normalization of
                      emojis and unusual usage of some punctuations
    replace_keyword_emojis: Replaces emojis whose name contains
                            keywords with that keyword
    remove_punctuations: 'keep_basic_punctuations' or 'all'
    reduce_punctuations: Convert multiple occurence of same
                         punctuation character to single character.
    use_lower_case: Convert text to lower case
    do_split_punctuation_text: Split punctutation from text
                               (required by some embeddings)
    Filtering:
    remove_sun_confusing_terms: Remove Tweets containing "Sun"
                                as it is commonly used to refer
                                to the British newspaper "The Sun"
                                or Sunday
    only_text_containing_keywords: Remove Tweets containing no
                                   element of `keywords`
    maximum_bounding_box_area: Removes Tweets with
                               bounding_box_area > maximum_bounding_box_area
    only_unique_text: Removes Tweets with duplicate text
    processes: Number of processes used to normalize text

    Returns
    Dataset with normalized text and original text
    """

    ds = normalize_text_dataset(
        ds,
        keywords=keywords,
        key_text_original=key_text_original,
        key_text_normalized=key_text_normalized,
        key_text_backup=key_text_backup,
        ignore_non_ascii=ignore_non_ascii,
        replace_keyword_emojis=replace_keyword_emojis,
        remove_punctuations=remove_punctuations,
        reduce_punctuations=reduce_punctuations,
        use_lower_case=use_lower_case,
        do_split_punctuation_text=do_split_punctuation_text,
        processes=processes,
    )

    ds = filter_text_dataset(
        ds,
        key_text=key_text_normalized,
        remove_sun_confusing_terms=remove_sun_confusing_terms,
        only_text_containing_keywords=only_text_containing_keywords,
        maximum_bounding_box_area=maximum_bounding_box_area,
        keywords=keywords,
        only_unique_text=only_unique_text,
    )
    if reset_index and "index" in ds.variables.keys():
        ds = a2.dataset.load_dataset.reset_index_coordinate(ds)
    return ds
