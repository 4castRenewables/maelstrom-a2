import a2.utils.utils
import numpy as np


def test_parallelize(vectors_for_cross_product, results_cross_product):
    a, b = vectors_for_cross_product
    kwargs_as_dict = dict(axis=None)
    args_zipped = zip(a, b)
    cross_product = a2.utils.utils.parallelize(
        function=np.cross,
        args_zipped=args_zipped,
        single_arg=False,
        kwargs_as_dict=kwargs_as_dict,
    )
    assert np.array_equal(cross_product, results_cross_product)


def test_timing(caplog):
    @a2.utils.utils.timing
    def f(x):
        return x

    f(1)
    for record in caplog.records:
        assert len(record.msg) > 40 and len(record.msg) < 50
