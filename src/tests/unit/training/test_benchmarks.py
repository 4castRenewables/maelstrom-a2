import a2.training.benchmarks
import a2.utils.testing


def test_timer(capsys):
    timer = a2.training.benchmarks.Timer(print_all_single_time_stats=True)
    timer.start(a2.training.benchmarks.TimeType.RUN)
    timer.end(a2.training.benchmarks.TimeType.RUN)
    timer.print_all_time_stats()
    captured = capsys.readouterr().out
    io_length = len(captured)
    assert io_length == 103
