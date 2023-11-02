import operator

import numpy as np


def build_vocab(texts):
    def split_test(s):
        return s.split()

    tweets = np.array(list(map(split_test, texts)))
    vocab = {}
    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def check_embeddings_coverage(texts, embeddings):
    vocab = build_vocab(texts)
    covered = {}
    oov = {}
    n_covered = 0
    n_oov = 0
    for word in vocab:
        if word == "":
            print("found it")
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
