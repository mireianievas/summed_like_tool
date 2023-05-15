from os.path import dirname
import glob
from pathlib import Path

from astropy.io import fits
from astropy.table import Table


def make_obs_index(path):
    """generate the OBS index for the DL3 files in path"""
    files = glob.glob(f"{path}")
    filedir = dirname(files[0])
    # empty lists with the columns to be stored
    obs_ids = []
    ra_pnts = []
    dec_pnts = []
    tstarts = []
    tstops = []
    dead_counts = []
    for _file in files:
        with fits.open(_file) as hdus:
            events_header = hdus[1].header
            obs_ids.append(events_header["OBS_ID"])
            ra_pnts.append(events_header["RA_PNT"])
            dec_pnts.append(events_header["DEC_PNT"])
            tstarts.append(events_header["TSTART"])
            tstops.append(events_header["TSTOP"])
            dead_counts.append(events_header["DEADC"])

    # create obs index
    obs_table = Table(
        [obs_ids, ra_pnts, dec_pnts, tstarts, tstops, dead_counts],
        names=("OBS_ID", "RA_PNT", "DEC_PNT", "TSTART", "TSTOP", "DEADC"),
        dtype=(">i8", ">f4", ">f4", ">f4", ">f4", ">f4"),
        meta={
            "name": "OBS_INDEX",
            "EXTNAME": "OBS_INDEX",
            "HDUDOC": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
            "HDUVERS": "0.2",
            "HDUCLASS": "GADF",
            "HDUCLAS1": "INDEX",
            "HDUCLAS2": "OBS",
        },
    )
    # set units
    obs_table["RA_PNT"].unit = "deg"
    obs_table["DEC_PNT"].unit = "deg"
    obs_table["TSTART"].unit = "s"
    obs_table["TSTOP"].unit = "s"
    obs_file = f"{filedir}/obs-index.fits.gz"
    obs_table.write(obs_file, overwrite=True)


def make_hdu_index(path):
    """generate the HDU index for the DL3 files in path"""
    files = glob.glob(f"{path}")
    filedir = dirname(files[0])
    # empty lists with the columns to be stored
    obs_ids = []
    hdu_types = []
    hdu_classes = []
    file_dirs = []
    file_names = []
    hdu_names = []
    # dictionary connecting the hdu types with their names
    hdu_types_dict = {
        "EVENTS": "events",
        "GTI": "gti",
        "EFFECTIVE AREA": "aeff",
        "ENERGY DISPERSION": "edisp",
        "POINT SPREAD FUNCTION": "psf",
        "RAD_MAX": "rad_max",
    }
    hdu_classes_dict = {
        "EVENTS": "events",
        "GTI": "gti",
        "EFFECTIVE AREA": "aeff_2d",
        "ENERGY DISPERSION": "edisp_2d",
        "POINT SPREAD FUNCTION": "psf_table",
        "RAD_MAX": "rad_max_2d",
    }
    # Test file type (contains RAD_MAX / PSF?)
    with fits.open(files[0]) as test_file:
        for htype in list(hdu_types_dict.keys()):
            try:
                test_file[htype]
            except KeyError:
                del hdu_types_dict[htype]
                del hdu_classes_dict[htype]
    
    for _file in files:
        with fits.open(_file) as hdus:
            for hdu in hdus[1:]:
                obs_ids.append(hdus[1].header["OBS_ID"])
                hdu_types.append(hdu_types_dict[hdu.name])
                hdu_classes.append(hdu_classes_dict[hdu.name])
                file_dirs.append("./")
                file_names.append(_file.split("/")[-1])
                hdu_names.append(hdu.name)
    hdu_table = Table(
        [obs_ids, hdu_types, hdu_classes, file_dirs, file_names, hdu_names],
        names=(
            "OBS_ID",
            "HDU_TYPE",
            "HDU_CLASS",
            "FILE_DIR",
            "FILE_NAME",
            "HDU_NAME",
        ),
        dtype=(">i8", "S30", "S30", "S100", "S50", "S30"),
        meta={
            "name": "OBS_INDEX",
            "EXTNAME": "OBS_INDEX",
            "HDUDOC": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
            "HDUVERS": "0.2",
            "HDUCLASS": "GADF",
            "HDUCLAS1": "INDEX",
            "HDUCLAS2": "OBS",
        },
    )

    hdu_file = f"{filedir}/hdu-index.fits.gz"
    hdu_table.write(hdu_file, overwrite=True)


def make_obs_hdu_index(path):
    """generate OBS and HDU index for the DL3 files in a directory"""
    make_obs_index(path)
    make_hdu_index(path)
