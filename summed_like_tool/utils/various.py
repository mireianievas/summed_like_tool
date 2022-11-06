from astropy.io import fits
import numpy as np
import io
import gzip


def read_fits_gz(f):
    ### Robuts fits.gz reader
    try:
        fc = fits.open(f)
    except OSError:
        # some gzipped fits files seem to have issues in astropy/ds9, therefore we first decompress.
        with gzip.open(f, "rb") as _f:
            gz_bytes = _f.read()
            io_bytes = io.BytesIO(gzip.decompress(gz_bytes))
            fc = fits.open(io_bytes)  # ,ignore_missing_simple=True)

    return fc


def closest_in_array(array, value, shift=0):
    loc = np.argmin(np.abs(array - value))
    if loc < 0:
        loc = np.max(0, loc + shift)
    elif loc > 0:
        loc = np.min(len(array) - 1, loc + shift)

    return (array[loc], loc)


def slice_in_mapaxis(mapaxis, value_min, value_max, padding=0):
    edges = mapaxis.edges
    loc1 = closest_in_array(edges, value_min, -padding)[1]
    loc2 = closest_in_array(edges, value_max, padding)[1]

    for loc in range(loc1, loc2 + 1, 1):
        try:
            newedges.append(mapaxis.slices(loc))
        except:
            newedges = mapaxis.slices(loc)

    return (newedges, loc1, loc2)
