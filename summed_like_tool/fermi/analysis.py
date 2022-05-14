import gzip
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
from gammapy.irf import EDispKernel,PSFMap
from gammapy.maps import MapAxis,Map
from gammapy.data import EventList
from gammapy.modeling.models import (
    SkyModel,
    TemplateSpatialModel,
    PowerLawNormSpectralModel,
    create_fermi_isotropic_diffuse_model,
)

from .files import Files
from ..utils.plotting import new_viridis
#self.lat_bute = Table.read(self.lat_bute_file,format='ascii')
#self.lat_ebin = Table.read(self.lat_ebin_file,format='ascii')


class InstrumentResponse(Files):
    def read_exposure(self):
        self.exposure  = Map.read(self.expmap_f)
    def read_psf(self):
        self.psf       = PSFMap.read(self.psf_f,format='gtpsf')
    def read_energy_dispersion(self):
        self.drmap = pyfits.open(self.edrm_f)
    def read_diffuse_background(self):
        self.diffgalac = Map.read(self.diffgal_f)
        self.diffiso   = create_fermi_isotropic_diffuse_model(
            filename=self.iso_f, 
            interp_kwargs={"fill_value": None}
        )
    def read_irfs(self):
        self.read_exposure()
        self.read_psf()
        self.read_energy_dispersion()
        self.read_diffuse_background()

class EnergyAxes(InstrumentResponse):
     def set_energy_axes(self):
        energy_lo = self.drmap['DRM'].data['ENERG_LO']*u.MeV
        energy_hi = self.drmap['DRM'].data['ENERG_HI']*u.MeV
        self.energy_axis      = MapAxis.from_energy_edges(np.append(energy_lo[0],energy_hi))
        self.energy_axis_true = self.energy_axis.copy(name="energy_true")
        #self.energy_edges = self.energy_axis.edges

class EnergyMatrix(EnergyAxes):
    def energy_dispersion_matrix(self):
        drm = self.drmap['DRM'].data['MATRIX']
        drm_matrix = np.array(list(drm))
        self.edisp_kernel = EDispKernel(
            axes=[self.energy_axis_true,
                  self.energy_axis],
            data=drm_matrix
        )
        
    def draw(self):
        print('Peek on original edisp kernel')
        return(self.edisp_kernel.peek())

class Events(EnergyAxes):
    def load_events(self):
        # Create a local file (important if gzipped, as sometimes it fails to read)
        with gzip.open(self.event_f) as gzfile:
            with open("temp_events.fits", "wb") as f:
                f.write(gzip.decompress(gzfile.read()))        
    
        #eventfits = read_fits_gz(event_f)
        self.eventfits = pyfits.open("temp_events.fits")
        self.events    = EventList.read("temp_events.fits")
    
    def get_src_skycoord(self):
        dsval2 = self.eventfits[1].header['DSVAL2']
        ra,dec = [float(k) for k in dsval2.split("(")[1].split(",")[0:2]]
        self.src_pos = SkyCoord(ra, dec, unit="deg", frame="fk5")
        
    def counts_map(self):
        self.countsmap = Map.create(
            skydir=self.src_pos,
            npix=(self.exposure.geom.npix[0][0], 
                  self.exposure.geom.npix[1][0]),
            proj="TAN",
            frame="fk5",
            binsz=(self.exposure.geom.pixel_scales)[0],
            axes=[self.energy_axis],
            dtype=float,
        )
        self.countsmap.fill_by_coord({"skycoord": self.events.radec,
                                      "energy":   self.events.energy
                                     })

    def draw_counts_map(self):
        f = plt.figure(dpi=120)
        percentiles = np.percentile(self.countsmap.sum_over_axes().smooth(1).data,
                                    [10,99.9])
        #print(percentiles)
        axes = self.countsmap.sum_over_axes().smooth(1.5).plot(stretch="sqrt", 
                                                    vmin=percentiles[0],
                                                    vmax=percentiles[1],
                                                    cmap=new_viridis())

        axes.grid(lw=0.5,color='white',alpha=0.5,ls='dotted')
        return(f)

class Analysis(Events,EnergyMatrix):
    def diffuse_background_models(self):
        
        # Doing a cutout may improve fitting speed.
        self.diffuse_cutout = self.diffgalac.cutout(self.countsmap.geom.center_skydir,
                                                    self.countsmap.geom.width[0])
        
        self.template_diffuse = TemplateSpatialModel(
            self.diffuse_cutout, normalize=False
        )
        
        self.diffgalac_cutout = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=self.template_diffuse,
            name="diffuse-iem",
        )
        
    def plot_diffuse_models(self):
        F = plt.figure(figsize=(4,3),dpi=150)
        plt.loglog()
        
        # Exposure varies very little with energy at these high energies
        energy = np.geomspace(0.1 * u.GeV, 1 * u.TeV, 20)
        dnde = self.template_diffuse.map.interp_by_coord(
            {"skycoord": self.src_pos, "energy_true": energy},
            method="linear",
            fill_value=None,
        )*u.Unit("1/(cm2*s*MeV*sr)")
        
        plt.plot(energy, dnde*u.sr, 
                 marker="*",
                 ls='dotted',
                 label='diffuse gal / sr')
        
        energy_range = [0.1, 2000] * u.GeV
        self.diffiso.spectral_model.plot(energy_range, 
                                        sed_type="dnde", 
                                        label='diffuse iso');

        plt.legend()
        return(F)
        
    #def 