from astropy.io import fits as pyfits
import io
import gzip


def read_fits_gz(f):
    ### Robuts fits.gz reader
    try:
        fc = pyfits.open(f)
    except OSError:
        # some gzipped fits files seem to have issues in astropy/ds9, therefore we first decompress.
        with gzip.open(f, "rb") as _f:
            gzbytes = _f.read()
            iobytes = io.BytesIO(gzip.decompress(gzbytes))
            fc = pyfits.open(iobytes)  # ,ignore_missing_simple=True)

    return fc


def closest_in_array(array, value, shift=0):
    loc = np.argmin(np.abs(array - value))
    if loc < 0:
        loc = np.max(0, loc + shift)
    elif loc > 0:
        loc = np.min(len(array) - 1, loc + shift)

    return (array[loc], loc)


def slice_in_mapaxis(mapaxis, valuemin, valuemax, padding=0):
    edges = mapaxis.edges
    loc1 = closest_in_array(edges, valumin, -padding)[1]
    loc2 = closest_in_array(edges, valumax, padding)[1]

    for loc in range(loc1, loc2 + 1, 1):
        try:
            newedges.append(mapaxis.slices(loc))
        except:
            newedges = mapaxis.slices(loc)

    return (newedges, loc1, loc2)
