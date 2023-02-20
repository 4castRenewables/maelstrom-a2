import abc

import pyproj


class UnitConverter(abc.ABC):
    @abc.abstractmethod
    def transform(self):
        pass


class ConvFromNGToLatLong(UnitConverter):
    def __init__(self):
        self.transformer = pyproj.Transformer.from_crs("epsg:27700", "epsg:4326")

    def transform(self, easting, northing, errcheck=True):
        """
        Converts units in British National Grid (easting, northing) to latitude, longitude
        Note, shift from east,north -> north,east convention
        Parameters:
        ----------
        easting: Distance from western origin in m
        northing: Distance from southern origin in m

        Returns
        -------
        latitude, longitude
        """
        return self.transformer.transform(easting, northing, errcheck=errcheck)


class ConvFromLatLongToNG(UnitConverter):
    def __init__(self):
        self.transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:27700")

    def transform(self, latitude, longitude, errcheck=True):
        """
        Converts units from latitude, longitude to British National Grid (easting, northing)
        Note, shift from north,east -> east,north convention
        Parameters:
        ----------
        easting: Distance from western origin in m
        northing: Distance from southern origin in m

        Returns
        -------
        easting, northing
        """
        return self.transformer.transform(latitude, longitude, errcheck=errcheck)
