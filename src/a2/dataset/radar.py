#!/usr/bin/python
"""
Extract data from UK Met Office Rain Radar NIMROD image files.
Parse NIMROD format image files, display header data and allow extraction of
raster image to an ESRI ASCII (.asc) format file. A bounding box may be
specified to clip the image to the area of interest. Can be imported as a
Python module or run directly as a command line script.
Author: Richard Thomas
Version: 2.00 (27 November 2021)
Public Repository: https://github.com/richard-thomas/MetOffice_NIMROD
Command line usage:
  python nimrod.py [-h] [-q] [-x] [-bbox XMIN XMAX YMIN YMAX] [infile] [outfile]
positional arguments:
  infile                (Uncompressed) NIMROD input filename
  outfile               Output raster filename (*.asc)
optional arguments:
  -h, --help            show this help message and exit
  -q, --query           Display metadata
  -x, --extract         Extract raster file in ASC format
  -bbox XMIN XMAX YMIN YMAX
                        Bounding box to clip raster data to
Note that any bounding box must be specified in the same units and projection
as the input file. The bounding box does not need to be contained by the input
raster but must intersect it.
Example command line usage:
  python nimrod.py -bbox 279906 285444 283130 290440
    -xq 200802252000_nimrod_ng_radar_rainrate_composite_1km_merged_UK_zip
    plynlimon_catchments_rainfall.asc
Example Python module usage:
    import nimrod
    a = nimrod.Nimrod(open(
        '200802252000_nimrod_ng_radar_rainrate_composite_1km_merged_UK_zip',
        'rb'))
    a.query()
    a.extract_asc(open('full_raster.asc', 'w'))
    a.apply_bbox(279906, 285444, 283130, 290440)
    a.query()
    a.extract_asc(open('clipped_raster.asc', 'w'))
Notes:
  1. Valid for v1.7 and v2.6-4 of NIMROD file specification
  2. Assumes image origin is top left (i.e. that header[24] = 0)
  3. Tested on UK and European composite 1km and 5km data, using Python 3.9
     in Windows 10
Copyright (c) 2021 Richard Thomas
(Nimrod.__init__() method based on read_nimrod.py by Charles Kilburn Aug 2008)
This program is free software: you can redistribute it and/or modify
it under the terms of the Artistic License 2.0 as published by the
Open Source Initiative (http://opensource.org/licenses/Artistic-2.0)
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
import array
import dataclasses
import gzip
import itertools
import logging
import os
import pathlib
import re
import struct
import typing as t
import warnings

import a2.dataset
import a2.utils
import numpy as np
import pandas as pd
import xarray

TYPE_KEY_TWEETS = object


@dataclasses.dataclass(frozen=True)
class KeyRadar:
    """Stores most common names for keys in radar netcdf datasets"""

    time: str = "time"
    tp: str = "tp_mm"
    longitude: str = "longitude"
    latitude: str = "latitude"
    x_ngt: str = "x_ngt"
    y_ngt: str = "y_ngt"


def get_key_radar(key_radar: t.Optional[KeyRadar], cumulative_radar: bool = False) -> KeyRadar:
    """Get version of KeyRadar depending on setup"""
    if key_radar is None:
        if cumulative_radar:
            key_radar = a2.dataset.radar.KeyRadar(tp="tp_mm_cum")
        else:
            key_radar = a2.dataset.radar.KeyRadar()
    return key_radar


class Nimrod:
    """Reading, querying and processing of NIMROD format rainfall data files."""

    class RecordLenError(Exception):
        """
        Exception Type: NIMROD record length read from file not as expected.
        """

        def __init__(self, actual: int, expected: int, location: str):
            self.message = "Incorrect record length %d bytes (expected %d) at %s." % (actual, expected, location)

    class HeaderReadError(Exception):
        """Exception Type: Read error whilst parsing NIMROD header elements."""

        pass

    class PayloadReadError(Exception):
        """Exception Type: Read error whilst parsing NIMROD raster data."""

        pass

    class BboxRangeError(Exception):
        """
        Exception Type: Bounding box specified out of range of raster image.
        """

        pass

    def __init__(
        self,
        filename: pathlib.Path | str = None,
        filename_zipped: pathlib.Path | str = None,
        file_content: t.TextIO = None,
        fake_data: bool = False,
    ):
        """
        Parse all header and data info from a NIMROD data file into this object.
        (This method based on read_nimrod.py by Charles Kilburn Aug 2008)
        Args:
            filename: filename of NIMROD binary file
        Raises:
            RecordLenError: NIMROD record length read from file not as expected
            HeaderReadError: Read error whilst parsing NIMROD header elements
            PayloadReadError: Read error whilst parsing NIMROD raster data
        """
        self.convert_from_ngt_to_latlong = a2.dataset.units.ConvFromNGToLatLong()
        self.filename_nimrod = filename

        def check_record_len(file: t.TextIO, expected: int, location: str):
            """
            Check record length in C struct is as expected.
            Args:
                file: file to read from
                expected: expected value of record length read
                location: description of position in file (for reporting)
            Raises:
                HeaderReadError: Read error whilst reading record length
                RecordLenError: Unexpected NIMROD record length read from file
            """

            # Unpack length from C struct (Big Endian, 4-byte long)
            try:
                (record_length,) = struct.unpack(">l", file.read(4))
            except Exception:
                raise Nimrod.HeaderReadError
            if record_length != expected:
                raise Nimrod.RecordLenError(record_length, expected, location)

        if filename is not None:
            file = open(filename, "rb")
        elif filename_zipped is not None:
            file = gzip.open(filename_zipped, "rb")
        else:
            file = file_content
        # Header should always be a fixed length record
        check_record_len(file, 512, "header start")

        try:
            # Read first 31 2-byte integers (header fields 1-31)
            gen_ints = array.array("h")
            gen_ints.fromfile(file, 31)
            gen_ints.byteswap()

            # Read next 28 4-byte floats (header fields 32-59)
            gen_reals = array.array("f")
            gen_reals.fromfile(file, 28)
            gen_reals.byteswap()

            # Read next 45 4-byte floats (header fields 60-104)
            spec_reals = array.array("f")
            spec_reals.fromfile(file, 45)
            spec_reals.byteswap()

            # Read next 56 characters (header fields 105-107)
            characters = array.array("B")
            characters.fromfile(file, 56)

            # Read next 51 2-byte integers (header fields 108-)
            spec_ints = array.array("h")
            spec_ints.fromfile(file, 51)
            spec_ints.byteswap()

        except Exception:
            file.close()
            raise Nimrod.HeaderReadError

        check_record_len(file, 512, "header end")

        # Extract strings and make duplicate entries to give meaningful names
        chars = characters.tobytes().decode()
        self.units = chars[0:8]
        self.data_source = chars[8:32]
        self.title = chars[32:55]

        # Store header values in a list so they can be indexed by "element
        # number" shown in NIMROD specification (starts at 1)
        self.hdr_element = [None]  # Dummy value at element 0
        self.hdr_element.extend(gen_ints)
        self.hdr_element.extend(gen_reals)
        self.hdr_element.extend(spec_reals)
        self.hdr_element.extend([self.units])
        self.hdr_element.extend([self.data_source])
        self.hdr_element.extend([self.title])
        self.hdr_element.extend(spec_ints)

        # related to date and time
        self.validity_year = self.hdr_element[1]
        self.validity_month = self.hdr_element[2]
        self.validity_day = self.hdr_element[3]
        self.validity_hour = self.hdr_element[4]
        self.validity_minute = self.hdr_element[5]
        self.validity_second = self.hdr_element[6]
        self.date_year = self.hdr_element[7]
        self.date_month = self.hdr_element[8]
        self.date_day = self.hdr_element[9]
        self.date_hour = self.hdr_element[10]
        self.date_minute = self.hdr_element[11]

        # Duplicate some of values to give more meaningful names
        self.nrows = self.hdr_element[16]
        self.ncols = self.hdr_element[17]
        self.n_data_specific_reals = self.hdr_element[22]
        self.n_data_specific_ints = self.hdr_element[23] + 1
        # Note "+ 1" because header value is count from element 109
        self.y_top = self.hdr_element[34]
        self.y_pixel_size = self.hdr_element[35]
        self.x_left = self.hdr_element[36]
        self.x_pixel_size = self.hdr_element[37]

        # Calculate other image bounds (note these are pixel centres)
        self.x_right = self.x_left + self.x_pixel_size * (self.ncols - 1)
        self.y_bottom = self.y_top - self.y_pixel_size * (self.nrows - 1)

        # Read payload (actual raster data)
        array_size = self.ncols * self.nrows
        check_record_len(file, array_size * 2, "data start")

        self.data = array.array("h")
        try:
            self.data.fromfile(file, array_size)
            self.data.byteswap()
        except Exception:
            file.close()
            raise Nimrod.PayloadReadError
        if fake_data:
            self.data = np.random.randint(0, 100, array_size)
        check_record_len(file, array_size * 2, "data end")
        file.close()

    def query(self):
        """Print complete NIMROD file header information."""

        print("NIMROD file raw header fields listed by element number:")
        print("General (Integer) header entries:")
        for i in range(1, 32):
            print(" ", i, "\t", self.hdr_element[i])
        print("General (Real) header entries:")
        for i in range(32, 60):
            print(" ", i, "\t", self.hdr_element[i])
        print("Data Specific (Real) header entries (%d):" % self.n_data_specific_reals)
        for i in range(60, 60 + self.n_data_specific_reals):
            print(" ", i, "\t", self.hdr_element[i])
        print("Data Specific (Integer) header entries (%d):" % self.n_data_specific_ints)
        for i in range(108, 108 + self.n_data_specific_ints):
            print(" ", i, "\t", self.hdr_element[i])
        print("Character header entries:")
        print("  105 Units:           ", self.units)
        print("  106 Data source:     ", self.data_source)
        print("  107 Title of field:  ", self.title)

        # Print out info & header fields
        # Note that ranges are given to the edge of each pixel
        print(
            "\nValidity Time:  %2.2d:%2.2d on %2.2d/%2.2d/%4.4d"
            % (
                self.hdr_element[4],
                self.hdr_element[5],
                self.hdr_element[3],
                self.hdr_element[2],
                self.hdr_element[1],
            )
        )
        print(
            "Easting range:  %.1f - %.1f (at pixel steps of %.1f)"
            % (
                self.x_left - self.x_pixel_size / 2,
                self.x_right + self.x_pixel_size / 2,
                self.x_pixel_size,
            )
        )
        print(
            "Northing range: %.1f - %.1f (at pixel steps of %.1f)"
            % (
                self.y_bottom - self.y_pixel_size / 2,
                self.y_top + self.y_pixel_size / 2,
                self.y_pixel_size,
            )
        )
        print("Image size: %d rows x %d cols" % (self.nrows, self.ncols))

    def _extract_data_and_header(self):
        """
        Convert data to header in dictionary format and
        data to 2D numpy array
        """

        if self.x_pixel_size != self.y_pixel_size:
            warnings.warn("x_pixel_size(%d) != y_pixel_size(%d)" % (self.x_pixel_size, self.y_pixel_size))

        # Write header to output file. Note that data is valid at the centre
        # of each pixel so "xllcenter" rather than "xllcorner" must be used
        attributes = {}
        attributes["ncols"] = self.ncols
        attributes["nrows"] = self.nrows
        attributes["xllcenter"] = self.x_left
        attributes["yllcenter"] = self.y_bottom
        attributes["cellsize"] = self.y_pixel_size
        attributes["nodata_value"] = self.hdr_element[3]
        attributes["Units"] = self.units
        attributes["Data source"] = self.data_source
        attributes["Title of field"] = self.title

        # Note that ranges are given to the edge of each pixel
        attributes["Validity Time"] = "%2.2d:%2.2d on %2.2d/%2.2d/%4.4d" % (
            self.hdr_element[4],
            self.hdr_element[5],
            self.hdr_element[3],
            self.hdr_element[2],
            self.hdr_element[1],
        )
        attributes["Eastingrange"] = "{:.1f} - {:.1f} (at pixel steps of {:.1f})".format(
            self.x_left - self.x_pixel_size / 2,
            self.x_right + self.x_pixel_size / 2,
            self.x_pixel_size,
        )
        attributes["Northingrange"] = "{:.1f} - {:.1f} (at pixel steps of {:.1f})".format(
            self.y_bottom - self.y_pixel_size / 2,
            self.y_top + self.y_pixel_size / 2,
            self.y_pixel_size,
        )
        attributes["mage size"] = "%d rows x %d cols" % (self.nrows, self.ncols)

        field_values = np.full((self.nrows, self.ncols), np.nan)
        for i in range(self.nrows):
            for j in range(self.ncols):
                field_values[i, j] = self.data[i * self.ncols + j]
        self.attributes = attributes
        self.field_numpy = field_values
        return attributes, field_values

    def _get_coordinates_in_ngt(self):
        """Create coordinates in units of the national grid."""
        self.x_ngt = np.linspace(self.x_left, self.x_right, self.ncols, endpoint=True)
        self.y_ngt = np.linspace(self.y_bottom, self.y_top, self.nrows, endpoint=True)
        if not np.all(np.diff(self.x_ngt) == self.x_pixel_size) or not np.all(np.diff(self.y_ngt) == self.y_pixel_size):
            raise ValueError(
                "Coordinates not spaced as expected: "
                f"{np.all(np.diff(self.x_ngt) == self.x_pixel_size)=} and/or"
                f"{np.all(np.diff(self.y_ngt) == self.y_pixel_size)=}!"
            )
        self.x_ngt, self.y_ngt = np.meshgrid(self.x_ngt, self.y_ngt)
        return self.x_ngt, self.y_ngt

    def _get_coordinates_in_lat_long(self):
        """Create coordinates in units latitude and longitude."""
        self.x_ngt, self.y_ngt = self._get_coordinates_in_ngt()
        # NOTE, order switches from east-west, north-south -> north-south, east-west!
        self.y_lat, self.x_long = self.convert_from_ngt_to_latlong.transform(self.x_ngt, self.y_ngt)

    def extract_ngt_and_tp_mm(self):
        """Create coordinates [national grid] and total precipitation."""
        self._extract_data_and_header()
        self._get_coordinates_in_ngt()
        self._compute_tp_mm()
        return self.x_ngt, self.y_ngt, self.field_tp_mm

    def extract_xarray_dataset(self, version: str = "default"):
        """Convert nimrod file to netcdf file."""
        self._extract_data_and_header()
        if version == "default":
            self._get_coordinates_in_lat_long()
        elif version == "minimal":
            self._get_coordinates_in_ngt()
        else:
            raise NotImplementedError(f"{version=} not implemented!")
        self._compute_tp_mm()
        if version == "default":
            self.ds = xarray.Dataset(
                data_vars=dict(
                    tp_mm_hr_Div32=(
                        ["time", "y", "x"],
                        np.expand_dims(self.field_numpy, 0),
                    ),
                    tp_mm=(["time", "y", "x"], np.expand_dims(self.field_tp_mm, 0)),
                ),
                coords=dict(
                    time=np.expand_dims(
                        np.datetime64(
                            f"{self.date_year:04}-{self.date_month:02}-{self.date_day:02}"
                            f"T{self.date_hour:02}:{self.date_minute:02}:00.000000000"
                        ),
                        0,
                    ),
                    latitude=(["y", "x"], self.y_lat),
                    longitude=(["y", "x"], self.x_long),
                    x_ngt=(["y", "x"], self.x_ngt),
                    y_ngt=(["y", "x"], self.y_ngt),
                ),
                attrs=self.attributes,
            )
        elif version == "minimal":
            self.ds = xarray.Dataset(
                data_vars=dict(
                    tp_mm=(["time", "y", "x"], np.expand_dims(self.field_tp_mm, 0)),
                ),
                coords=dict(
                    time=np.expand_dims(
                        np.datetime64(
                            f"{self.date_year:04}-{self.date_month:02}-{self.date_day:02}"
                            f"T{self.date_hour:02}:{self.date_minute:02}:00.000000000"
                        ),
                        0,
                    ),
                    x_ngt=(["y", "x"], self.x_ngt),
                    y_ngt=(["y", "x"], self.y_ngt),
                ),
                attrs=self.attributes,
            )
        return self.ds

    def _compute_tp_mm(self):
        """Convert total precipitation from code units to total precipitation
        accumulated within time range of file (5 min) in mm."""
        self.field_tp_mm = np.full_like(self.field_numpy, np.nan)
        mask = self.field_numpy != -1
        # data files contain integer precipitation rates in unit of (mm/hr)*32
        # convert from code to (mm) within 5 minutes (time interval of files)
        self.field_tp_mm[mask] = self.field_numpy[mask] / 32 * 5.0 / 60.0

    def to_netcdf(self, filename: str | pathlib.Path):
        """
        Write netcdf file to be read with xarray
        Args:
            filename: name of file data will be save to
        """
        self.extract_xarray_dataset().to_netcdf(filename)


def _convert_file_to_xarray_dataset(
    filename_nimrod: str | pathlib.Path = None, file_content_nimrod: t.TextIO = None
) -> xarray.Dataset:
    """Create xarray dataset from nimrod file"""
    nim = Nimrod(filename=filename_nimrod, file_content=file_content_nimrod)
    return nim.extract_xarray_dataset()


def _convert_file(filename: str | pathlib.Path = None):
    """Create xarray dataset from zipped nimrod file"""
    with gzip.open(filename, "rb") as file_content:
        try:
            return _convert_file_to_xarray_dataset(file_content_nimrod=file_content)
        except Exception as e:
            warnings.warn(f"Couldn't open nimrod file {filename}!\n{e}")
            return None


def convert_daily_files_to_multiple_netcdf(
    file_list: list[str | pathlib.Path], file_name_netcdf: str | pathlib.Path, chunk_size: int = 20
):
    """
    Convert list of daily files to multiple netcdf files

    Parameters:
    ----------
    file_list: List of daily filenames (nimrod files)
    file_name_netcdf: Prefix for filename of newly created netcdf file
    chunk_size: Number of files per chunk

    Returns
    -------
    """
    # sort file list as otherwise subsequent merge throws:
    # ValueError: Resulting object does not have monotonic global indexes along dimension time
    def match_date_rain_radar_file(filename):
        try:
            return int(
                re.findall(
                    r"^metoffice-c-band-rain-radar_uk_(\d+)_1km-composite.dat.gz",
                    filename,
                )[0]
            )
        except IndexError as e:
            raise ValueError(f"Couldn't parse: {filename}!") from e

    file_list = a2.utils.file_handling.sort_filenames_by_pattern(file_list, match_date_rain_radar_file)
    for i, partial_files in enumerate(a2.utils.utils.chunks(file_list, chunk_size)):
        ds_to_merge = a2.utils.utils.parallelize(_convert_file, partial_files, single_arg=True, processes=1)
        ds_to_merge = [x for x in ds_to_merge if x is not None]
        if len(ds_to_merge):
            xarray.merge(ds_to_merge).to_netcdf(file_name_netcdf.with_suffix(f".{i}.nc"))
        else:
            warnings.warn(f"No netcdf files {ds_to_merge=} found to merge in {partial_files=}!")


def _nimrod_daily_tarball_to_netcdf(filename_tarball: str | pathlib.Path, chunk_size: int = 10) -> pathlib.Path:
    """
    Convert files in single tarball (files for day each) to netcdf

    Parameters:
    ----------
    filename_tarball: Filename of (daily) tarball
    chunk_size: Number of files per chunk saved to single netcdf

    Returns
    -------
    """
    filename_tarball = pathlib.Path(filename_tarball)
    folder_minute_data = filename_tarball.with_suffix("")
    os.makedirs(folder_minute_data, exist_ok=True)
    a2.utils.file_handling.decompress_tarball(filename_tarball, folder_minute_data)
    file_name_netcdf = folder_minute_data / (folder_minute_data.name + ".nc")
    convert_daily_files_to_multiple_netcdf(
        a2.utils.file_handling.get_all_files(folder_minute_data / "*.dat.gz"),
        file_name_netcdf,
        chunk_size=chunk_size,
    )
    return file_name_netcdf


def all_nimrod_daily_tarball_to_netcdf(
    base_folder: t.Optional[str | pathlib.Path] = None,
    file_list: t.Optional[list[str | pathlib.Path]] = None,
    chunk_size: int = 10,
    processes: int = -1,
) -> list[str | pathlib.Path]:
    """
    Convert list of tarballs to netcdf files

    Parameters:
    ----------
    base_folder: Base folder to look for tarballs including its subdirectories
    file_list: List of tarballs of nimrod files
    chunk_size: Number of files per chunk saved to single netcdf
    processes: Number of tarballs saved to netcdf in parallel

    Returns
    -------
    """
    if file_list is None and base_folder is not None:
        file_list = a2.utils.file_handling.get_all_files(
            f"{base_folder.__str__()}/*dat.gz.tar",
            include_subdirectories=True,
        )
    logging.info(f"... converting {len(file_list)} files to netcdf")
    print(f"converting: {file_list=}")
    folder_list = a2.utils.utils.parallelize(
        _nimrod_daily_tarball_to_netcdf,
        zip(file_list),
        kwargs_as_dict={"chunk_size": chunk_size},
        processes=processes,
    )
    return folder_list


def _combine_netcdf(folder_list: list[str | pathlib.Path], filename: str | pathlib.Path) -> bool:
    """Save list of netcdf files to single file and return success status"""
    success = True
    print(f"_combine_netcdf: merging {folder_list=} to {filename=}")
    try:
        a2.dataset.load_dataset.save_dataset(
            xarray.open_mfdataset(folder_list, combine_attrs="drop_conflicts"), filename
        )
    except ValueError as e:
        warnings.warn(f"Couldn't combine netcdf files {folder_list} to single netcdf file {filename}!\n{e}")
        success = False
    return success


def _get_netcdf_filenames_from_tarball_folders(tarball_folders: list[str]) -> list[str]:
    """Create filenames of netcdf files per tarball in list of tarball folder"""
    folders_daily = []
    for tar in tarball_folders:
        folders_daily.append(tar.replace(".tar", ""))
    nc_files_per_folder = []
    for folder in folders_daily:
        match_files = os.path.join(folder, os.path.split(folder)[1] + ".[0-9]*.nc")
        files_in_folder = a2.utils.file_handling.get_all_files(match_files)
        nc_files_per_folder.append(files_in_folder)
    return nc_files_per_folder


def merge_nc_files_on_daily_basis(
    base_folder: str, input_is_folders: bool = False, remove_files: bool = False, processes: int = -1
):
    """
    Merge all netcdf files in subfolders of base_folder to single netcdf per day.
    This assumes the internal filename convention for nimrod filenames and
    that all all netcdf filenames describing a single day are found in a single folder.

    Parameters:
    ----------
    base_folder: Subfolders are searched for pattern "**/*.dat.gz.tar" following naming convention,
                 alternatively subfolders can be given see `input_is_folders`
    input_is_folders: Is `base_folder` a list of folders containing netcdf files to be merged
    remove_files: Single files deleted after merging
    processes: Number of files merged to single file in parallel

    Returns
    -------
    """
    if not input_is_folders:
        tarballs = a2.utils.file_handling.get_all_files(
            os.path.join(base_folder.__str__(), "**/*.dat.gz.tar"), recursive=True
        )
    else:
        tarballs = base_folder
    nc_files_per_folder = _get_netcdf_filenames_from_tarball_folders(tarballs)
    print(f"{nc_files_per_folder=}")
    a2.utils.utils.parallelize(
        _merge_nc_files_on_daily_basis,
        zip(nc_files_per_folder),
        single_arg=True,
        kwargs_as_dict={"remove_files": remove_files},
        processes=processes,
    )


def _merge_nc_files_on_daily_basis(files, remove_files=False):
    """
    See `merge_nc_files_on_daily_basis`. Function to be parallelized.
    """
    if not len(files):
        logging.info(f"Nothing to merge {files=}")
        return
    filename_netcdf = re.sub(r".dat.gz.\d+.nc", r".dat.gz.nc", files[0])
    success = _combine_netcdf(files, filename_netcdf)
    logging.info(f"merging files into {filename_netcdf}")
    if success and remove_files:
        for f in files:
            os.remove(f)


def from_ngt_to_ds_index(
    ds_radar: xarray.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    x_key: str = "x_ngt",
    y_key: str = "y_ngt",
    asserting: bool = False,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Convert x and y values (in ngt) to indices in `ds_radar` of respective fields

    Parameters:
    ----------
    ds_radar: Radar data as xarray dataset
    x: Values along west-east in national grid units
    y: Values along north-south in national grid units
    x_key: Key in dataset to corresponding west-east coordinate values in national grid units
    y_key: Key in dataset to corresponding north-south coordinate values in national grid units
    asserting: Additional (costly) consistency checks are done

    Returns
    indices for x, indices for y
    -------
    """
    y_shape, x_shape = ds_radar[x_key].shape
    x_min = ds_radar[x_key].min().item()
    x_max = ds_radar[x_key].max().item()
    x_spacing = (x_max - x_min) / (x_shape - 1)

    y_min = ds_radar[y_key].min().item()
    y_max = ds_radar[y_key].max().item()
    y_spacing = (y_max - y_min) / (y_shape - 1)

    x_index = (x - x_min) / x_spacing
    y_index = (y - y_min) / y_spacing
    x_index, y_index = x_index.astype(int), y_index.astype(int)

    if asserting:
        ds_test = ds_radar.sel(x=x_index, y=y_index)
        unique_values_x = np.sort(np.unique(ds_test[x_key].values))
        unique_values_y = np.sort(np.unique(ds_test[y_key].values))
        x_unique = np.sort(np.unique(x))
        y_unique = np.sort(np.unique(y))
        if not np.array_equal(unique_values_x, x_unique) or not np.array_equal(unique_values_y, y_unique):
            for u, s in zip(unique_values_x, x_unique):
                print(f"{u=}, {s=}")
            for u, s in zip(unique_values_y, y_unique):
                print(f"{u=}, {s=}")
            raise ValueError(
                f"Indices not correctly identified: {unique_values_x=} "
                f"should be {x_unique=} and {unique_values_y=} should be {y_unique=}"
            )
    return x_index, y_index


def get_time_ds_index(
    ds_radar: xarray.Dataset, times: list, key_time: str = "time", check_for_outliers: bool = True
) -> np.ndarray:
    """
    Convert date time values to indices in `ds_radar` of respective time field.

    Parameters:
    ----------
    ds_radar: Radar data as xarray dataset
    times: Time values
    key_time: Key in dataset to corresponding date time
    check_for_outliers: Additional consistency checks are done

    Returns
    indices for x, indices for y
    -------
    """
    times_int = times.astype(int)
    times_ds_int = ds_radar[key_time].values.astype(int)
    time_shape = times_ds_int.shape[0]
    if time_shape == 1:
        if not all(ds_radar[key_time] == t for t in times):
            raise ValueError(
                f"Single time present in dataset {ds_radar[key_time]} doesn't correspond to desired value {times}!"
            )
        else:
            return np.zeros(len(times), dtype=int)
    time_min = times_ds_int.min()
    time_max = times_ds_int.max()
    time_spacing = (time_max - time_min) / (time_shape - 1)
    mask = np.logical_and(times_int >= time_min, times_int <= time_max)
    if check_for_outliers and times_int[~mask].shape[0]:
        raise ValueError(f"Found times outide of dataset range: {times_int[~mask]=},{time_min=},{time_max=}")
    return ((times_int - time_min) / time_spacing).astype(int)


def assign_ngt_to_tweets(
    ds_tweets: xarray.Dataset,
    key_tweets=None,
) -> xarray.Dataset:
    """Compute coordinates in units of the national grid from latitude and longitude
    and save them to the dataset"""
    ds_tweets = ds_tweets.copy()
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    ds_tweets = a2.dataset.load_dataset.reset_index_coordinate(ds_tweets)
    converter_from_latong_to_ngt = a2.dataset.units.ConvFromLatLongToNG()
    results = a2.utils.utils.parallelize(
        converter_from_latong_to_ngt.transform,
        zip(
            ds_tweets[key_tweets.latitude].values,
            ds_tweets[key_tweets.longitude].values,
        ),
    )
    easting, northing = np.array(results).T
    ds_tweets[key_tweets.x_ngt] = (["index"], easting)
    ds_tweets[key_tweets.y_ngt] = (["index"], northing)
    return ds_tweets


def assign_radar_to_tweets_from_netcdf(
    ds_tweets: xarray.Dataset,
    list_radar_filenames: t.Optional[list[str | pathlib.Path]] = None,
    base_folder: t.Optional[str | pathlib.Path] = None,
    key_tweets: TYPE_KEY_TWEETS | None = None,
    key_radar: KeyRadar = KeyRadar(),
    round_ngt_offset: int = 500,
    round_ngt_decimal: int = -3,
    round_time_to_base: int = 5,
) -> xarray.Dataset:
    """
    Assign radar precipitation data that is stored in netcdf files to Tweets dataset.
    Tweets coordinates (latitude, longitude, time) are converted
    and rounded to value convention in radar dataset.

    Parameters:
    ----------
    ds_tweets: Dataset of Tweets
    list_radar_filenames: List of netcdf files storing radar data
    base_folder: Base folder to look for to look for netcdf files storing radar data (*.nc)
    key_tweets: Dataclass storing key convention for tweets datasets
    key_radar: Dataclass storing key convention for radar datasets
    round_ngt_offset: Round spatial coordinates of Tweets to this offset
    round_ngt_decimal: Round spatial coordinates of Tweets to decimal point
    round_time_to_base: Round time coordinate of Tweets to this base

    Returns
    dataset with added radar data
    -------
    """
    if not list_radar_filenames:
        radar_filenames = a2.utils.file_handling.get_all_files(
            os.path.join(base_folder.__str__(), "**/*.nc"), recursive=True
        )
    else:
        radar_filenames = list_radar_filenames
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    ds_tweets = _prepare_tweet_ds_for_radar(
        ds_tweets,
        key_tweets=key_tweets,
        round_ngt_offset=round_ngt_offset,
        round_ngt_decimal=round_ngt_decimal,
        round_time_to_base=round_time_to_base,
    )

    ds_tweets[key_tweets.tp_radar] = (
        ["index"],
        np.full(ds_tweets.index.shape[0], np.nan),
    )
    for radar_filename in radar_filenames:
        ds_radar = xarray.open_dataset(radar_filename)
        print(f"{ds_radar=}")
        radar_times = ds_radar[key_radar.time].values
        radar_times_min = radar_times.min()
        radar_times_max = radar_times.max()
        mask = (ds_tweets[key_tweets.time_radar] >= radar_times_min) & (
            ds_tweets[key_tweets.time_radar] <= radar_times_max
        )
        print(f"{radar_times=}")
        indices_select = ds_tweets["index"].loc[mask]

        tweets_x_ngt_rounded = ds_tweets[key_tweets.x_ngt_rounded].sel(index=indices_select).values
        tweets_y_ngt_rounded = ds_tweets[key_tweets.y_ngt_rounded].sel(index=indices_select).values
        time = ds_tweets[key_tweets.time_radar].sel(index=indices_select).values

        radar_values_sel = _get_tp_from_ds_radar(key_radar, ds_radar, tweets_x_ngt_rounded, tweets_y_ngt_rounded, time)
        ds_tweets[key_tweets.tp_radar].loc[dict(index=indices_select)] = radar_values_sel
    return ds_tweets


def _get_tp_from_ds_radar(
    key_radar: KeyRadar,
    ds_radar: xarray.Dataset,
    tweets_x_ngt_rounded: np.ndarray,
    tweets_y_ngt_rounded: np.ndarray,
    time: np.datetime64,
    asserting_xy: bool = False,
) -> np.ndarray:
    """
    Based on spatial coordinates (in units of the national grid) and time obtain total precipitation from radar dataset.

    Parameters:
    ----------
    key_radar: Dataclass storing key convention for radar datasets
    ds_radar: Dataset of Radar data
    tweets_x_ngt_rounded: Coordinates along west-east in units of the national grid for the Tweets
    tweets_y_ngt_rounded: Coordinates along north-south in units of the national grid for the Tweets
    time: Time values for Tweets
    asserting_xy: Additional (costly) consistency checks on spatial coordinates indices are done

    Returns
    -------
    radar values
    """
    index_x, index_y = from_ngt_to_ds_index(
        ds_radar, tweets_x_ngt_rounded, tweets_y_ngt_rounded, asserting=asserting_xy
    )
    index_time = get_time_ds_index(ds_radar, time, key_time=key_radar.time)
    dims_required = ("time", "y", "x")
    dims_found = ds_radar[key_radar.tp].coords.dims
    if dims_found != dims_required:
        raise IndexError(
            f"Require dims of precipitation {key_radar.tp} in radar ds to be {dims_required} but found {dims_found}"
        )
    radar_values_sel = ds_radar[key_radar.tp].values[index_time, index_y, index_x]
    return radar_values_sel


def _prepare_tweet_ds_for_radar(
    ds_tweets: xarray.Dataset,
    key_tweets: TYPE_KEY_TWEETS = None,
    round_ngt_offset: int = 500,
    round_ngt_decimal: int = -3,
    round_time_to_base: int = 5,
) -> xarray.Dataset:
    """
    Prepare Tweets dataset by converting latitude, longitude to
    national grid units at consistent intervals with radio data.
    In addition, round time to same convention as radar data.

    Parameters:
    ----------
    ds_tweets: Dataset of Tweets
    key_tweets: Dataclass storing key convention for tweets datasets
    round_ngt_offset: Round spatial coordinates of Tweets to this offset
    round_ngt_decimal: Round spatial coordinates of Tweets to decimal point
    round_time_to_base: Round time coordinate of Tweets to this base

    Returns
    dataset with prepared coordinates for radar data assignment
    -------
    """
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    ds_tweets = assign_ngt_to_tweets(
        ds_tweets,
        key_tweets=key_tweets,
    )
    ds_tweets[key_tweets.x_ngt_rounded] = (
        ["index"],
        a2.utils.utils.round_offset(
            ds_tweets[key_tweets.x_ngt].values,
            decimal=round_ngt_decimal,
            offset=round_ngt_offset,
        ),
    )
    ds_tweets[key_tweets.y_ngt_rounded] = (
        ["index"],
        a2.utils.utils.round_offset(
            ds_tweets[key_tweets.y_ngt].values,
            decimal=round_ngt_decimal,
            offset=round_ngt_offset,
        ),
    )
    ds_tweets[key_tweets.time_radar] = (
        ["index"],
        a2.utils.utils.vetorize_time_to_base_minutes(ds_tweets[key_tweets.time].values, base=round_time_to_base),
    )

    return ds_tweets


def assign_radar_to_tweets(
    ds_tweets: xarray.Dataset,
    key_tweets: TYPE_KEY_TWEETS = None,
    round_ngt_offset: int = 500,
    round_ngt_decimal: int = -3,
    round_time_to_base: int = 5,
    path_to_dapceda: str | pathlib.Path | None = None,
    processes: int = -1,
) -> xarray.Dataset:
    """
    Assign radar data from raw radar data files to Tweets dataset.
    Radar files are unzipped and preprocessed for this.

    Parameters:
    ----------
    ds_tweets: Dataset of Tweets
    key_tweets: Dataclass storing key convention for tweets datasets
    round_ngt_offset: Round spatial coordinates of Tweets to this offset
    round_ngt_decimal: Round spatial coordinates of Tweets to decimal point
    round_time_to_base: Round time coordinate of Tweets to this base
    path_to_dapceda: Path to "badc" folder downloaded as downloaded form CEDA archive
    processes: Number of radar files assigned to Tweets in parallel

    Returns
    dataset with added radar data
    -------
    """
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    if key_tweets is None:
        key_tweets = key_tweets()
    if path_to_dapceda is None:
        path_to_dapceda = a2.utils.file_handling.get_folder_radar()
    path_to_dapceda = pathlib.Path(path_to_dapceda)
    if not path_to_dapceda.is_dir():
        raise ValueError(f"Couldn't find folder {path_to_dapceda=}!")
    ds_tweets = _prepare_tweet_ds_for_radar(
        ds_tweets,
        key_tweets=key_tweets,
        round_ngt_offset=round_ngt_offset,
        round_ngt_decimal=round_ngt_decimal,
        round_time_to_base=round_time_to_base,
    )

    ds_tweets[key_tweets.tp_radar] = (
        ["index"],
        np.full(ds_tweets.index.shape[0], np.nan),
    )
    ds_tweets[key_tweets.time_radar_int] = (["index"], ds_tweets[key_tweets.time_radar].values.astype(int))
    ds_tweets, time_xngt_yngt = a2.dataset.utils_dataset.divide_ds_by_unique_values(
        ds_tweets,
        key_divide_by=key_tweets.time_radar_int,
        keys_values=[key_tweets.time_radar, key_tweets.x_ngt_rounded, key_tweets.y_ngt_rounded],
    )

    radar_tp_mm = a2.utils.utils.parallelize(
        get_tp_from_single_ds_radar,
        zip(*time_xngt_yngt),
        single_arg=False,
        kwargs_as_dict={
            "path_to_dapceda": path_to_dapceda,
        },
        processes=processes,
    )

    radar_tp_mm = np.array(a2.utils.utils.flatten_list(radar_tp_mm))
    ds_tweets[key_tweets.tp_radar] = (["index"], radar_tp_mm)
    return ds_tweets


def get_tp_from_single_ds_radar(
    time,
    tweets_x_ngt_rounded,
    tweets_y_ngt_rounded,
    path_to_dapceda=None,
    key_radar=KeyRadar(),
) -> np.ndarray:
    """
    Retrieve precipitation from single radar files at various locations.
    Finds respective binary file for time, converts it to an xarray dataset and returns precipitation
    at the desired locations.

    Parameters:
    ----------
    time: Time stamps for the desired coordinates (should all be same)
    tweets_x_ngt_rounded: Coordinates along west-east in units of the national grid for the Tweets
    tweets_y_ngt_rounded: Coordinates along north-south in units of the national grid for the Tweets
    path_to_dapceda: Path to "badc" folder downloaded as downloaded form CEDA archive
    key_radar: Dataclass storing key convention for radar datasets

    Returns
    radar values at given locations
    -------
    """
    if not all(time[0] == x for x in time):
        raise Exception(f"Time data not same time {time=}!")
    radar_filename_zipped, filename_tarball, folder_to_decompress = _get_filename_from_time(path_to_dapceda, time[0])
    if not radar_filename_zipped.is_file() and not folder_to_decompress.is_dir() and filename_tarball.is_file():
        a2.utils.file_handling.decompress_tarball(filename_tarball, folder_to_decompress)
    if not radar_filename_zipped.is_file():
        warnings.warn(f"Cannot find file {radar_filename_zipped}, skipping it ....")
        return np.full(np.shape(time), -np.inf)
    try:
        nim = a2.dataset.radar.Nimrod(filename_zipped=radar_filename_zipped)
    except (Nimrod.RecordLenError, Nimrod.HeaderReadError, Nimrod.PayloadReadError, Nimrod.BboxRangeError) as e:
        warnings.warn(f"{e}: Cannot convert file {radar_filename_zipped=}, skipping it ....")
        return np.full(np.shape(time), -np.inf)

    ds_radar = nim.extract_xarray_dataset(version="minimal")
    radar_values_sel = _get_tp_from_ds_radar(
        key_radar,
        ds_radar,
        np.array(tweets_x_ngt_rounded),
        np.array(tweets_y_ngt_rounded),
        np.array(time),
    )
    return radar_values_sel


def _get_filename_from_time(
    path_to_dapceda: pathlib.Path, time: np.datetime64
) -> t.Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Determine radar filename, folder based on timestamp"""
    year, month, day, hour, minute = a2.utils.times.time_breakdown(time)
    radar_filename_zipped = (
        path_to_dapceda / f"badc/ukmo-nimrod/data/composite/uk-1km/{year}/"
        f"metoffice-c-band-rain-radar_uk_{year}{month:02}{day:02}_1km-composite.dat.gz/"
        f"metoffice-c-band-rain-radar_uk_{year}{month:02}{day:02}{hour:02}{minute:02}_1km-composite.dat.gz"
    )
    filename_tarball = (
        path_to_dapceda / f"badc/ukmo-nimrod/data/composite/uk-1km/{year}/"
        f"metoffice-c-band-rain-radar_uk_{year}{month:02}{day:02}_1km-composite.dat.gz.tar"
    )
    folder_to_decompress = (
        path_to_dapceda / f"badc/ukmo-nimrod/data/composite/uk-1km/{year}/"
        f"metoffice-c-band-rain-radar_uk_{year}{month:02}{day:02}_1km-composite.dat.gz"
    )
    return radar_filename_zipped, filename_tarball, folder_to_decompress


def nimrod_ds_from_time(path_to_dapceda: pathlib.Path, time: np.datetime64, version: str = "default") -> xarray.Dataset:
    """Retrieve nimrod file based on time and return it as xarray.Dataset.
    `version` determines format of returned Dataset"""
    path_to_dapceda = pathlib.Path(path_to_dapceda)
    radar_filename_zipped, filename_tarball, folder_to_decompress = _get_filename_from_time(path_to_dapceda, time)
    if not radar_filename_zipped.is_file() and not folder_to_decompress.is_dir() and filename_tarball.is_file():
        a2.utils.file_handling.decompress_tarball(filename_tarball, folder_to_decompress)
    try:
        nim = a2.dataset.radar.Nimrod(filename_zipped=radar_filename_zipped)
    except (
        Nimrod.RecordLenError,
        Nimrod.HeaderReadError,
        Nimrod.PayloadReadError,
        Nimrod.BboxRangeError,
        FileNotFoundError,
    ) as e:
        warnings.warn(f"{e}: Cannot convert file {radar_filename_zipped=}, skipping it ....")
        return None
    ds_radar = nim.extract_xarray_dataset(version=version)
    return ds_radar


def nimrod_ds_cumulative_from_time(
    path_to_dapceda: pathlib.Path,
    time: np.datetime64,
    time_delta: float = 1,
    time_delta_units: str = "h",
    version: str = "default",
) -> xarray.Dataset:
    """
    Compute total precipitation by summing cumulative over time period `time_delta` [`time_delta_units`] up to `time`.

    Parameters:
    ----------
    path_to_dapceda: Path to "badc" folder downloaded as downloaded form CEDA archive
    time: Precipitation summed up to this time
    time_delta: Precipitation summed over this period of time
    time_delta_units: Units of `time_delta`
    version: Determines format of returned Dataset
             'default': Returns coordinates as longitude/latitude and in northing/easting of national grid
             'minimal': Ony returns coordinates as longitude/latitude

    Returns
    dataset with cumulated precipitation
    -------
    """
    coords = None
    ds_not_none = None
    time_increment_files = pd.Timedelta("5m")
    time_end = pd.to_datetime(time)
    time_start = pd.to_datetime(time) - pd.Timedelta(f"{time_delta}{time_delta_units}") + time_increment_files
    ds = nimrod_ds_from_time(path_to_dapceda, time_start, version=version)
    tp_list = []
    tp_list, coords, ds_not_none = _append_to_list(ds, tp_list, coords, ds_not_none)
    t = time_start + time_increment_files
    while t <= time_end:
        ds = nimrod_ds_from_time(path_to_dapceda, t)
        tp_list, coords, ds_not_none = _append_to_list(ds, tp_list, coords, ds_not_none)
        t = t + time_increment_files
    if not len(tp_list):
        warnings.warn(
            f"Found no files at {time=} with {time_delta=}, "
            f"{time_delta_units=} for cumulative nimrod ds, returning `None`"
        )
        return None
    coords_time = np.expand_dims(
        np.datetime64(time_end),
        0,
    )
    data_vars = dict(
        tp_mm_cum=(["time", "y", "x"], np.sum(tp_list, axis=0)),
    )
    return a2.dataset.utils_dataset.construct_dataset(ds_not_none, data_vars, time=coords_time)


def _append_to_list(ds, tp_list, coords, ds_not_none):
    """Helper function to append list with precipitation values"""
    if ds is not None:
        tp_list.append(ds.tp_mm.values)
        coords = ds.coords
        ds_not_none = ds
    return tp_list, coords, ds_not_none


def time_series_from_files(
    time_start: np.datetime64,
    time_end: np.datetime64,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    path_to_dapceda: str | pathlib.Path = None,
    time_delta: int = 1,
    time_delta_units: str = "h",
    key_radar: KeyRadar = None,
    key_time: str = "time",
    processes: int = -1,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Compute time series of cumulative radar precipitation for location given in long/lat.

    Parameters:
    ----------
    time_start: Time series starting time
    time_end: Time series final time
    longitudes: Longitudes of locations of interest
    latitudes: Latitudes of locations of interest
    path_to_dapceda: Path to "badc" folder downloaded as downloaded form CEDA archive
    time_delta: Precipitation summed over this period of time
    time_delta_units: Units of `time_delta`
    key_radar: Dataclass storing key convention for radar datasets
    key_time: Key in dataset to corresponding date time
    processes: Number of times computed in parallel

    Returns
    times, cumulative precipitation with shape: [times, lat/long, 0/1]
    -------
    """
    if path_to_dapceda is None:
        path_to_dapceda = a2.utils.constants.PATH_DAPCEDA
    if key_radar is None:
        key_radar = KeyRadar(tp="tp_mm_cum")
    time_increment = pd.Timedelta(f"{time_delta}{time_delta_units}")
    if not time_increment % pd.Timedelta("5m") == pd.Timedelta(0):
        raise ValueError(f"{time_delta}{time_delta_units} should be divisible by time spacing of files (5 min)!")
    time_end = a2.utils.utils.round_time_to_base_minutes(time_end, base=5)
    time_end = pd.to_datetime(time_end)
    time_start = a2.utils.utils.round_time_to_base_minutes(pd.to_datetime(time_start), base=5)
    time_start = pd.to_datetime(time_start)
    logging.info(
        f"Computing time series from {time_start=} to {time_end=} in increments of {time_delta}{time_delta_units}"
    )
    t = time_start
    converter_from_latong_to_ngt = a2.dataset.units.ConvFromLatLongToNG()
    x_ngt, y_ngt = np.array(converter_from_latong_to_ngt.transform(latitudes, longitudes))
    x_ngt_rounded = a2.utils.utils.round_offset(x_ngt, decimal=-3, offset=500)
    y_ngt_rounded = a2.utils.utils.round_offset(y_ngt, decimal=-3, offset=500)

    times = []
    while t <= time_end:
        times.append(t)
        t = t + time_increment
    times_tp = a2.utils.utils.parallelize(
        _get_values_single_time,
        zip(times, itertools.repeat(x_ngt_rounded), itertools.repeat(y_ngt_rounded)),
        processes=processes,
        single_arg=False,
        kwargs_as_dict=dict(
            path_to_dapceda=path_to_dapceda,
            time_delta=time_delta,
            time_delta_units=time_delta_units,
            key_radar=key_radar,
            key_time=key_time,
        ),
    )
    times_radar = [t[0] for t in times_tp]
    tp_radar = [t[1] for t in times_tp]
    return np.array(times_radar), np.array(tp_radar)


def _get_values_single_time(
    time: np.datetime64,
    x_ngt_rounded: np.ndarray,
    y_ngt_rounded: np.ndarray,
    path_to_dapceda: str | pathlib.Path = None,
    time_delta: int = 1,
    time_delta_units: str = "h",
    key_radar: KeyRadar = None,
    key_time: str = "time",
) -> t.Tuple[np.datetime64, np.ndarray]:
    """
    Obtains precipitation values from single file for all given locations.

    Parameters:
    ----------
    time: Time to get precipitation values
    x_ngt_rounded: Coordinates along west-east in units of the national grid for the Tweets
    y_ngt_rounded: Coordinates along north-south in units of the national grid for the Tweets
    path_to_dapceda: Path to "badc" folder downloaded as downloaded form CEDA archive
    time_delta: Precipitation summed over this period of time
    time_delta_units: Units of `time_delta`
    key_radar: Dataclass storing key convention for radar datasets
    key_time: Key in dataset to corresponding date time

    Returns
    -------
    time, list of radar precipitation shape: [long/lat]
    """
    if path_to_dapceda is None:
        path_to_dapceda = a2.utils.constants.PATH_DAPCEDA
    ds = nimrod_ds_cumulative_from_time(
        path_to_dapceda, time, time_delta=time_delta, time_delta_units=time_delta_units, version="minimal"
    )
    if ds is None:
        return time, np.full_like(x_ngt_rounded, np.nan)
    tp_list = []
    for x, y in zip(x_ngt_rounded, y_ngt_rounded):
        tp = _get_tp_from_ds_radar(key_radar, ds, x, y, np.array([time], dtype=np.datetime64), asserting_xy=True)
        tp_list.append(tp)
    return ds[key_time].values[0], np.array(tp_list)
