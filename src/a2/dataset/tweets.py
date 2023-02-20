import dataclasses


@dataclasses.dataclass(frozen=True)
class KeyTweets:
    """Dataclass storing key convention for tweets datasets"""

    x_ngt: str = "x_ngt"
    y_ngt: str = "y_ngt"
    x_ngt_rounded: str = "x_ngt_rounded"
    y_ngt_rounded: str = "y_ngt_rounded"
    time: str = "created_at"
    time_radar: str = "time_radar"
    time_stations: str = "time_half"
    time_radar_int: str = "time_radar_int"
    latitude: str = "latitude"
    latitude_stations: str = "station_latitude"
    longitude: str = "longitude"
    longitude_stations: str = "station_longitude"
    tp_radar: str = "tp_mm_radar"
    station_tp: str = "station_tp_mm"
    distance_stations: str = "station_distance_km"
