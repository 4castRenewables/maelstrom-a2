import pathlib

import a2.plotting
import a2.utils
import distutils
import numpy as np
import pytest


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"
BASELINE_DIR = DATA_FOLDER / "baseline/"


def setup_directory(tmp_path, name_subdirectory):
    directory = tmp_path / name_subdirectory / "badc/"
    directory.mkdir(parents=True)
    folder_tarball_origin = DATA_FOLDER / "radar/badc"
    distutils.dir_util.copy_tree(folder_tarball_origin.__str__(), directory.__str__())
    return directory


def test_plot_precipiation_map(
    fake_dataset_precipitation,
    fake_dataset_tweets_weather_map,
    fake_weather_maps,
):
    axes, plots = a2.plotting.weather_maps.plot_precipiation_map(
        ds_precipitation=fake_dataset_precipitation,
        ds_tweets=fake_dataset_tweets_weather_map,
        key_latitude="latitude",
        key_longitude="longitude",
        key_time="time_half",
        key_tp="tp_h",
        return_plots=True,
    )
    plot_values = []
    for p in np.ndarray.flatten(plots):
        plot_values.append(a2.plotting.utils_plotting._get_values_from_pcolormesh_object(p))
    plot_values = np.array(plot_values)
    assert np.allclose(plot_values, fake_weather_maps)


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_radar_map_with_tweets(tmp_path, fake_dataset_tweets_radar_stations_plotting):
    setup_directory(tmp_path, "test_plot_radar_map_with_tweets")
    fig = a2.plotting.weather_maps.plot_radar_map_with_tweets(
        ds=fake_dataset_tweets_radar_stations_plotting,
        grid_shape=(1, 3),
        figsize=None,
        choice_type="increment_time",
        selection_delta_time=5,
        selection_delta_time_units="m",
        selection_key_twitter_time="time_radar",
        selector_use_limits=[True, False],
        xlim=None,
        ylim=None,
        path_to_dapceda=tmp_path / "test_plot_radar_map_with_tweets/",
        increment_time_value=np.datetime64("2020-02-07T00:00:00.000000000"),
        increment_time_delta=5,
        increment_time_delta_units="m",
        vmax=0.1,
        cumulative_radar=False,
        processes=1,
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_radar_map_with_tweets_cumulative_limits(tmp_path, fake_dataset_tweets_radar_stations_plotting):
    setup_directory(tmp_path, "test_plot_radar_map_with_tweets_cumulative_limits")
    fig = a2.plotting.weather_maps.plot_radar_map_with_tweets(
        ds=fake_dataset_tweets_radar_stations_plotting,
        grid_shape=(1, 3),
        figsize=None,
        choice_type="increment_time",
        selection_delta_time=5,
        selection_delta_time_units="m",
        selection_key_twitter_time="time_radar",
        selector_use_limits=[True, True],
        xlim=None,
        ylim=None,
        path_to_dapceda=tmp_path / "test_plot_radar_map_with_tweets_cumulative_limits/",
        increment_time_value=np.datetime64("2020-02-07T00:00:00.000000000"),
        increment_time_delta=5,
        increment_time_delta_units="m",
        vmax=0.1,
        cumulative_radar=True,
        cumulative_delta_time=5,
        cumulative_delta_time_units="m",
        processes=1,
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_radar_map_with_tweets_limits_upper(tmp_path, fake_dataset_tweets_radar_stations_plotting):
    setup_directory(tmp_path, "test_plot_radar_map_with_tweets_cumulative_limits")
    fig = a2.plotting.weather_maps.plot_radar_map_with_tweets(
        ds=fake_dataset_tweets_radar_stations_plotting,
        grid_shape=(1, 3),
        figsize=None,
        choice_type="increment_time",
        selection_delta_time=5,
        selection_delta_time_units="m",
        selection_key_twitter_time="time_radar",
        selector_use_limits=[False, True],
        xlim=None,
        ylim=None,
        path_to_dapceda=tmp_path / "test_plot_radar_map_with_tweets_cumulative_limits/",
        increment_time_value=np.datetime64("2020-02-07T00:00:00.000000000"),
        increment_time_delta=5,
        increment_time_delta_units="m",
        vmax=0.1,
        cumulative_radar=False,
        processes=1,
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_tp_station_tweets(tmp_path, fake_tweets_stations_plotting, fake_weather_stations_plotting):
    setup_directory(tmp_path, "test_plot_tp_station_tweets")
    a2.utils.testing.print_copy_paste_fake_dataset(fake_tweets_stations_plotting)
    fig = a2.plotting.weather_maps.plot_tp_station_tweets(
        ds=fake_tweets_stations_plotting,
        df_stations=fake_weather_stations_plotting,
        grid_shape=(1, 3),
        colormap="tab20c",
        vmin=0,
        vmax=1,
        fontsize=14,
        selection_delta_time=5,
        selection_delta_time_units="m",
        choice_type="increment_time",
        increment_time_value=np.datetime64("2020-02-07T00:30:00.000000000"),
        increment_time_delta=5,
        increment_time_delta_units="m",
        processes=1,
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_radar_from_time(tmp_path):
    setup_directory(tmp_path, "test_plot_radar_from_time")
    fig, ds = a2.plotting.weather_maps.plot_radar_from_time(
        time=np.datetime64("2020-02-07T00:05:00.000000000"),
        path_to_dapceda=tmp_path / "test_plot_radar_from_time/",
        cumulative=False,
        time_delta=5,
        time_delta_units="m",
        vmin=None,
        vmax=None,
        ax=None,
        xlim=None,
        ylim=None,
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_radar_from_time_cumulative(tmp_path):
    setup_directory(tmp_path, "test_plot_radar_from_time")
    fig, ds = a2.plotting.weather_maps.plot_radar_from_time(
        time=np.datetime64("2020-02-07T00:05:00.000000000"),
        path_to_dapceda=tmp_path / "test_plot_radar_from_time/",
        cumulative=True,
        time_delta=5,
        time_delta_units="m",
        vmin=None,
        vmax=None,
        ax=None,
        xlim=None,
        ylim=None,
    )
    return fig
