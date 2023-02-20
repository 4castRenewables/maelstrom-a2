import a2.preprocess
import a2.utils
import numpy as np
import pytest_cases


@pytest_cases.parametrize(
    "dataset, expected_sorted_oov, expected_vocab_coverage, expected_text_coverage, expected_vocab",
    [
        (
            "results_normalize_filter",
            [],
            1,
            1,
            {",": 1, "?": 1, "hi": 1, "is": 1, "it": 1, "raining": 1},
        ),
        (
            "dataset_less_optimal_coverage",
            [("NOTPARTOFVOCAB", 1)],
            0.8571428571428571,
            0.9,
            {
                ",": 1,
                "?": 1,
                "NOTPARTOFVOCAB": 1,
                "hi": 1,
                "is": 2,
                "it": 2,
                "raining": 2,
            },
        ),
    ],
)
def test_check_embeddings_coverage(
    dataset,
    expected_sorted_oov,
    expected_vocab_coverage,
    expected_text_coverage,
    expected_vocab,
    request,
):
    ds = request.getfixturevalue(dataset)
    embeddings = a2.preprocess.embedding.load_glove_embeddings()
    (
        sorted_oov,
        vocab_coverage,
        text_coverage,
        vocab,
    ) = a2.preprocess.embedding.check_embeddings_coverage(ds.text_normalized.values, embeddings)
    assert sorted_oov == expected_sorted_oov
    assert np.isclose(vocab_coverage, expected_vocab_coverage)
    assert text_coverage == expected_text_coverage
    assert vocab == expected_vocab
