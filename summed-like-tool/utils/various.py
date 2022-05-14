from astropy.io import fits as pyfits
import io
import gzip

def read_fits_gz(f):
    ### Robuts fits.gz reader
    try:
        fc = pyfits.open(f)
    except OSError:
        # some gzipped fits files seem to have issues in astropy/ds9, therefore we first decompress.
        with gzip.open(f, 'rb') as _f:
            gzbytes = _f.read()
            iobytes = io.BytesIO(gzip.decompress(gzbytes))
            fc = pyfits.open(iobytes)#,ignore_missing_simple=True)
        
    return(fc)

