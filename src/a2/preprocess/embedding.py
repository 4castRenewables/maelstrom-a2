import logging
import operator
import pathlib
import typing as t

import numpy as np


FILE_LOC = pathlib.Path(__file__).parent
EMBEDDINGS_FOLDER = FILE_LOC / "../../../data/embeddings/"
TWITTER_EMBEDDINGS_PATH = EMBEDDINGS_FOLDER / "glove.twitter.27B/"


def load_glove_embeddings(
    twitter_style: bool = True,
    twitter_vector_dimensions: int = 25,
) -> t.Dict:
    """
    Load Glove embeddings

    See https://nlp.stanford.edu/projects/glove/ for more info.
    Parameters:
    ----------
    twitter_style: Use embeddings trained on Twitter based corpus
                   else load love.840B.300d trained on Common Crawl
    twitter_vector_dimensions: Length of vectors representation
                               of tokens 25, 50, 100, 200 available

    Returns
    -------
    glove embeddings as dictionary
    """
    if twitter_style:
        path = TWITTER_EMBEDDINGS_PATH / f"glove.twitter.27B.{twitter_vector_dimensions}d.txt"
        glove_embeddings = {}
        with open(
            path,
            encoding="utf8",
        ) as f:
            for line in f:
                values = line.split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype="float64")
                glove_embeddings[word] = embedding
    else:
        path = EMBEDDINGS_FOLDER / "glove.840B.300d.pkl/glove.840B.300d.pkl"
        glove_embeddings = np.load(path, allow_pickle=True)
    logging.info("Loaded %s word vectors." % len(glove_embeddings))
    return glove_embeddings


def build_vocabulary(texts: t.Sequence) -> t.Dict:
    """
    Build vocabulary based on words in sequence of give text

    'Words' are obtained by splitting texts at whitespaces
    Parameters:
    ----------
    texts: Sequence of texts
    Returns
    -------
    dictionary; keys: words in vocab, values: occurence of words in texts
    """

    def generate_tokens(text):
        return text.split()

    texts_tokenized = np.array(list(map(generate_tokens, texts)), dtype=object)
    vocab: t.Dict[str, int] = {}
    for tt in texts_tokenized:
        for token in tt:
            try:
                vocab[token] += 1
            except KeyError:
                vocab[token] = 1
    return vocab


def check_embeddings_coverage(texts, embeddings):
    """
    Check coverage of words in texts in given set of embeddings

    Returns:
        sorted_oov: Sorted dictionary with occurence of
                    words that are not part of the embedding's vocobulary
        vocab_coverage: Percentage of individual words covered
                        by embedding's vocobulary
        text_coverage: Percentage of total text covered by
                       embedding's vocobulary
        vocab: Dictionary of vocab of texts including occurence of terms
    Parameters:
    ----------
    texts: Sequence of texts
    embeddings: Dictionary of embeddings,
                e.g. `a2.preprocess.embedding.load_glove_embeddings`
    Returns
    -------
    sorted_oov, vocab_coverage, text_coverage, vocab
    """
    vocab = build_vocabulary(texts)
    covered = {}
    oov = {}
    n_covered = 0
    n_oov = 0
    for word in vocab:
        if word == "":
            logging.info("found it")
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except KeyError:
            oov[word] = vocab[word]
            n_oov += vocab[word]
    vocab_coverage = len(covered) / len(vocab)
    text_coverage = n_covered / (n_covered + n_oov)
    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_oov, vocab_coverage, text_coverage, vocab
