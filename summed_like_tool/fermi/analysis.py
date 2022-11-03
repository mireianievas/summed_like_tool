import gzip
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
from gammapy.datasets import MapDataset
from gammapy.irf import (
    EDispKernel,
    PSFMap,
    EDispMap,
    EDispKernelMap,
)

from gammapy.maps import MapAxis, Map, WcsGeom, RegionGeom
from gammapy.data import EventList
from gammapy.modeling.models import (
    SkyModel,
    Models,
    TemplateSpatialModel,
    PowerLawNormSpectralModel,
    EBLAbsorptionNormSpectralModel,
    create_fermi_isotropic_diffuse_model,
)

from gammapy.modeling.models.core import DatasetModels

from .skymodel import FermiSkyModel
from .files import Files
from ..utils.plotting import new_viridis

# self.lat_bute = Table.read(self.lat_bute_file,format='ascii')
# self.lat_ebin = Table.read(self.lat_ebin_file,format='ascii')


class InstrumentResponse(Files):
    def read_exposure(self):
        self.log.info("Reading exposure")
        self.exposure = Map.read(self.expmap_f)

    def read_psf(self):
        self.log.info("Reading PSF")
        self.psf = PSFMap.read(self.psf_f, format="gtpsf")

    def read_energy_dispersion(self):
        self.log.info("Reading exposure")
        self.drmap = pyfits.open(self.edrm_f)

    def read_diffuse_background(self):
        self.log.info("Reading galactic diffuse")
        self.diffgalac = Map.read(self.diffgal_f)
        self.log.info("Reading isotropic diffuse")
        self.diffiso = create_fermi_isotropic_diffuse_model(
            filename=self.iso_f, interp_kwargs={"fill_value": None}
        )
        self.diffiso._name = "{}-{}".format(self.diffiso.name, self.tag)

    def read_irfs(self):
        self.read_exposure()
        self.read_psf()
        self.read_energy_dispersion()
        self.read_diffuse_background()


class EnergyAxes(InstrumentResponse):
    def set_energy_axes(self):
        self.log.info("Setting energy axes")
        energy_lo = self.drmap["DRM"].data["ENERG_LO"] * u.MeV
        energy_hi = self.drmap["DRM"].data["ENERG_HI"] * u.MeV
        self.energy_axis = MapAxis.from_energy_edges(np.append(energy_lo[0], energy_hi))
        self.energy_axis_true = self.energy_axis.copy(name="energy_true")
        # self.energy_edges = self.energy_axis.edges


class EnergyMatrix(EnergyAxes):
    def energy_dispersion_matrix(self):
        self.log.info("Creating energy dispersion kernel")
        drm = self.drmap["DRM"].data["MATRIX"]
        drm_matrix = np.array(list(drm))
        self.edisp_kernel = EDispKernel(
            axes=[self.energy_axis_true, self.energy_axis], data=drm_matrix
        )

    def plot_energy_dispersion_matrix(self):
        print("Peek on original edisp kernel")
        return self.edisp_kernel.peek()


class Events(EnergyAxes):
    def load_events(self):
        self.log.info("Loading events")
        # Create a local file (important if gzipped, as sometimes it fails to read)
        try:
            with gzip.open(self.events_f) as gzfile:
                with open("temp_events.fits", "wb") as f:
                    f.write(gzip.decompress(gzfile.read()))

            # eventfits = read_fits_gz(event_f)
            self.eventfits = pyfits.open("temp_events.fits")
            self.events = EventList.read("temp_events.fits")
        except:
            self.eventfits = pyfits.open(self.events_f)
            self.events = EventList.read(self.events_f)

    def get_src_skycoord(self):
        self.log.info("Loading sky coordinates")
        try:
            dsval2 = self.eventfits[1].header["DSVAL2"]
            ra, dec = [float(k) for k in dsval2.split("(")[1].split(",")[0:2]]
        except IndexError:
            history = str(self.eventfits[1].header["HISTORY"])
            ra, dec = (
                history.split("angsep(RA,DEC,")[1]
                .replace("\n", "")
                .split(")")[0]
                .split(",")
            )

        self.src_pos = SkyCoord(ra, dec, unit="deg", frame="fk5")

    def counts_map(self):
        self.log.info("Creating counts map")
        self.countsmap = Map.create(
            skydir=self.src_pos,
            npix=(self.exposure.geom.npix[0][0], self.exposure.geom.npix[1][0]),
            proj="TAN",
            frame="fk5",
            binsz=(self.exposure.geom.pixel_scales)[0],
            axes=[self.energy_axis],
            dtype=float,
        )
        self.countsmap.fill_by_coord(
            {"skycoord": self.events.radec, "energy": self.events.energy}
        )

    def plot_counts_map(self):
        f = plt.figure(dpi=120)
        percentiles = np.percentile(
            self.countsmap.sum_over_axes().smooth(1).data, [10, 99.9]
        )
        # print(percentiles)
        axes = (
            self.countsmap.sum_over_axes()
            .smooth(1.5)
            .plot(
                stretch="sqrt",
                vmin=percentiles[0],
                vmax=percentiles[1],
                cmap=new_viridis(),
            )
        )

        axes.grid(lw=0.5, color="white", alpha=0.5, ls="dotted")
        # self.CountsMapFigure = f
        # self.CountsMapAxes = axes
        # return(f)


class FermiAnalysis(Events, EnergyMatrix):
    def set_targetname(self, targetname):
        self.targetname = targetname

    def diffuse_background_models(self):

        self.log.info("Creating galactic diffuse model cutout")

        # Doing a cutout may improve fitting speed.
        self.diffuse_cutout = self.diffgalac.cutout(
            self.countsmap.geom.center_skydir, self.countsmap.geom.width[0]
        )

        self.template_diffuse = TemplateSpatialModel(
            self.diffuse_cutout, normalize=False
        )

        self.diffgalac_cutout = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=self.template_diffuse,
            name="diffuse-iem",
        )

    def plot_diffuse_models(self):
        F = plt.figure(figsize=(4, 3), dpi=150)
        plt.loglog()

        # Exposure varies very little with energy at these high energies
        energy = np.geomspace(0.1 * u.GeV, 1 * u.TeV, 20)
        dnde = self.template_diffuse.map.interp_by_coord(
            {"skycoord": self.src_pos, "energy_true": energy},
            method="linear",
            fill_value=None,
        ) * u.Unit("1/(cm2*s*MeV*sr)")

        plt.plot(energy, dnde * u.sr, marker="*", ls="dotted", label="diffuse gal / sr")

        energy_range = [0.1, 2000] * u.GeV
        self.diffiso.spectral_model.plot(
            energy_range, sed_type="dnde", label="diffuse iso"
        )

        plt.legend()
        # return(F)

    def plot_psf_containment(self):
        f = plt.figure(figsize=(6.5, 3), dpi=130)
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        self.psf.plot_psf_vs_rad(ax=ax1)
        self.psf.plot_containment_radius_vs_energy(ax=ax2, linewidth=2)
        plt.tight_layout()

    def plot_exposure_interpolate(self):
        self.exposure_interp = self.exposure.interp_to_geom(
            self.countsmap.geom.as_energy_true
        )
        f = plt.figure(figsize=(10, 3))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        self.exposure.slice_by_idx({"energy_true": 3}).plot(ax=ax1, add_cbar=True)
        self.exposure_interp.slice_by_idx({"energy_true": 0}).plot(
            ax=ax2, add_cbar=True
        )

    def set_edisp_interpolator(self):
        drmap = pyfits.open(self.edrm_f)
        energy_lo = drmap["DRM"].data["ENERG_LO"] * u.MeV
        energy_hi = drmap["DRM"].data["ENERG_HI"] * u.MeV
        drm = drmap["DRM"].data["MATRIX"]
        drm_matrix = np.array(list(drm))
        drm_eaxis = MapAxis.from_energy_edges(np.append(energy_lo[0], energy_hi))
        drm_eaxis_true = drm_eaxis.copy(name="energy_true")
        edisp_kernel = EDispKernel(axes=[drm_eaxis_true, drm_eaxis], data=drm_matrix)

        axis_reco = MapAxis.from_edges(
            self.countsmap.geom.axes["energy"].edges,
            name="energy",
            unit="MeV",
            interp="log",
        )

        axis_true = axis_reco.copy(name="energy_true")

        energy_reco, energy_true = np.meshgrid(axis_true.center, axis_reco.center)

        drm_interp = edisp_kernel.evaluate(
            "linear", **{"energy": energy_reco, "energy_true": energy_true}
        )

        self.edisp_interp_kernel = EDispKernel(
            axes=[axis_true, axis_reco], data=np.asarray(drm_interp)
        )

    def set_fake_edisp_interpolator(self):
        # For fake diagonal response (in case no EDISP is available)
        e_true = self.exposure_interp.geom.axes["energy_true"]
        edisp = EDispMap.from_diagonal_response(energy_axis_true=e_true)
        edisp_map = edisp.edisp_map

    def set_ebl_absorption_from_model(self, ebl_absorption=None):
        self.ebl_absorption = ebl_absorption

    def set_ebl_absorption_from_redshift(self, redshift, model=None):
        if model == None:
            model = "dominguez"

        self.ebl_absorption = EBLAbsorptionNormSpectralModel.read_builtin(
            model, redshift=redshift
        )

    def create_skymodel(self):
        self.log.info("Creating full skymodel")
        self.SkyModel = FermiSkyModel(self.xml_f)
        self.SkyModel.set_target_name(self.targetname)
        self.SkyModel.set_galdiffuse(self.diffuse_cutout)
        self.SkyModel.set_isodiffuse(self.diffiso)
        self.SkyModel.set_ebl_absorption(self.ebl_absorption)
        self.SkyModel.create_full_skymodel()

    def add_source_to_exclusion_region(self, src=None, radius=0.1 * u.deg, reset=False):

        try:
            self.exclusion_regions
            assert reset == False
        except:
            self.exclusion_regions = []

        if src != None:
            if isinstance(src, str):
                exclusion_region = CircleSkyRegion(
                    center=SkyCoord.from_name("Crab Nebula", frame="galactic"),
                    radius=radius,
                )
            elif isinstance(src, SkyCoord):
                exclusion_region = CircleSkyRegion(
                    center=src.galactic,
                    radius=radius,
                )
            self.exclusion_regions.append(exclusion_region)

    def add_exclusion_region(self, region):
        try:
            self.exclusion_regions
            assert reset == False
        except:
            self.exclusion_regions = []
        self.exclusion_regions.append(region)

    def create_exclusion_mask(self):
        skydir = self.src_pos.galactic
        excluded_geom = self.countsmap.geom.copy()
        if len(self.exclusion_regions) == 0:
            self.log.info("Creating empty/dummy exclusion region")
            pos = SkyCoord(0, 90, unit="deg")
            exclusion_region = CircleSkyRegion(pos, 0.00001 * u.deg)
            self.exclusion_mask = ~excluded_geom.region_mask([exclusion_region])
        else:
            self.log.info("Creating exclusion region")
            self.exclusion_mask = ~excluded_geom.region_mask(self.exclusion_regions)

    def create_dataset(self):
        self.log.info("Creating Mapdataset (self.dataset)")
        try:
            mask_safe = self.exclusion_mask
        except:
            mask_bool = np.zeros(self.countsmap.geom.data_shape).astype("bool")
            mask_safe = Map.from_geom(self.countsmap.geom, mask_bool)
            mask_safe.data = np.asarray(mask_safe.data == 0, dtype=bool)

        self.dataset = MapDataset(
            models=Models(self.SkyModel.list_sources),
            counts=self.countsmap,
            exposure=self.exposure_interp,
            psf=self.psf,
            edisp=EDispKernelMap.from_edisp_kernel(self.edisp_interp_kernel),
            mask_safe=mask_safe,
            name="Fermi-LAT_{}".format(self.unique_name),
        )

    def gen_analysis(
        self, targetname, ebl_absorption=None, redshift=None, ebl_model=None
    ):
        self.read_irfs()
        self.set_energy_axes()
        self.energy_dispersion_matrix()
        self.load_events()
        self.get_src_skycoord()
        self.counts_map()
        self.diffuse_background_models()
        self.set_edisp_interpolator()
        self.set_targetname(targetname)
        if ebl_absorption != None:
            self.set_ebl_absorption_from_model(ebl_absorption)
        elif redshift != None:
            self.set_ebl_absorption_from_redshift(redshift, ebl_model)
        self.create_skymodel()
        self.create_dataset()

    def gen_plots(self):
        self.plot_energy_dispersion_matrix()
        self.plot_counts_map()
        self.plot_diffuse_models()
        self.plot_psf_containment()
        self.plot_exposure_interpolate()

    def link_parameters_to_analysis(self, Analysis):
        self.log.info("Linking parameters to target Analysis object")
        NewDatasetModel = []
        for k, model in enumerate(Analysis.dataset.models):
            if "fermi-diffuse-iso" in model.name:
                # Link parameters of the second component only,
                # the first one is the shape of the ISO which depends on event type.
                self.dataset.models[k].spectral_model.model2 = Analysis.dataset.models[
                    k
                ].spectral_model.model2
                NewDatasetModel.append(self.dataset.models[k])
            else:
                # Link the entire model
                NewDatasetModel.append(Analysis.dataset.models[k])

        self.dataset.models = DatasetModels(NewDatasetModel)
