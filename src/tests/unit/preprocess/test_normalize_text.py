import a2.preprocess
import a2.utils
import pytest_cases
import xarray


@pytest_cases.parametrize(
    "to_normalize, remove_punctuations, result",
    [
        ("fake_dataset_to_normalize_filter", "keep_basic_punctuations", "results_normalize_filter"),
        ("fake_dataset_to_normalize_filter", "all", "results_normalize_filter_no_punctuation"),
    ],
)
def test_normalize_filter_dataset(to_normalize, remove_punctuations, result, request):
    ds = request.getfixturevalue(to_normalize)
    ds_results_normalize_filter = request.getfixturevalue(result)
    ds_normalized_filtered = a2.preprocess.normalize_text.normalize_filter_dataset(
        ds,
        keywords=None,
        reset_index=True,
        key_text_original="text",
        key_text_normalized="text_normalized",
        key_text_backup="text_original",
        ignore_non_ascii=True,
        replace_keyword_emojis=True,
        remove_punctuations=remove_punctuations,
        reduce_punctuations=True,
        use_lower_case=True,
        do_split_punctuation_text=True,
        remove_sun_confusing_terms=True,
        only_text_containing_keywords=True,
        maximum_bounding_box_area=100,
        only_unique_text=True,
        processes=1,
    )
    xarray.testing.assert_equal(ds_normalized_filtered, ds_results_normalize_filter)
