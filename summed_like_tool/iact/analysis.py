import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import PointSkyRegion, CircleSkyRegion

# from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
#    SpectrumDatasetOnOff,
#    FluxPointsDataset,
)
# from gammapy.estimators import FluxPointsEstimator,LightCurveEstimator
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.maps import WcsGeom, MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel
from gammapy.visualization import plot_spectrum_datasets_off_regions

from .files import Files
from ..utils.datastore import make_obs_hdu_index


class InstrumentResponse(Files):
    pass


class Observations(Files):
    def create_datastore(self, overwrite=False):
        self.log.info("Creating datastore from {}".format(self.dl3_path))
        try:
            if overwrite:
                raise OSError
            self.datastore = DataStore.from_dir(self.dl3_path)
        except OSError:
            self.log.warning(
                "obs/hdu index files not found, creating them from {}".format(
                    self.dl3_path
                )
            )
            make_obs_hdu_index(self.dl3_path)
            self.datastore = DataStore.from_dir(self.dl3_path)

    def get_target_name(self):
        self.target_name = np.unique(self.datastore.obs_table["OBJECT"])[0]
        self.log.info("The source is {}".format(self.target_name))

    def set_target_name(self, name):
        self.target_name = name
        self.log.info("The source is {}".format(self.target_name))

    def get_observations(self):
        self.obs_list = self.datastore.obs_table["OBS_ID"].data
        self.observations_total = self.datastore.get_observations(
            self.obs_list,
            required_irf=["aeff", "edisp", "rad_max"],
            skip_missing=False,
        )


class EnergyAxes(Files):
    def set_energy_axes(self, en_edges=None, en_min=None, en_max=None, nbins=None):
        self.log.info("Setting energy axes")

        if en_edges == None:
            if en_min == None:
                en_min = 10**1 * u.GeV
            if en_max == None:
                en_max = 10**5 * u.GeV
            if nbins == None:
                nbins = int(np.log10(en_max / en_min) * 5)  # 5 bins per decade
            reco_energy_edges = np.geomspace(en_min / u.GeV, en_max / u.GeV, nbins) * u.GeV
            true_energy_edges = np.geomspace(en_min / u.GeV, en_max / u.GeV, nbins) * u.GeV
        else:
            reco_energy_edges = en_edges
            true_energy_edges = en_edges

        self.energy_axis = MapAxis.from_energy_edges(reco_energy_edges)
        self.energy_axis_true = MapAxis.from_energy_edges(true_energy_edges).copy(
            name="energy_true"
        )


class Analysis1D(Observations, EnergyAxes):
    def set_src_pos(self, src_pos=None, target_name=None):
        if src_pos != None:
            self.src_pos = src_pos
        else:
            if target_name == None:
                target_name = self.target_name
            self.src_pos = SkyCoord.from_name(target_name)
        self.log.info("Setting source position to {}".format(self.src_pos))

    def set_on_region(self):
        self.log.info("Setting on region and empty dataset template")
        self.on_region = PointSkyRegion(self.src_pos)
        ### Hack to allow for Fermi+IACT fit (otherwise pointskyregion.contains returns None)
        self.on_region.meta = {"include": False}
        geom = RegionGeom.create(region=self.on_region, axes=[self.energy_axis])
        self.dataset_template = SpectrumDataset.create(
            geom=geom, energy_axis_true=self.energy_axis_true
        )

    def run_region_finder(self, n_off_regions=1):
        self.log.info("Setting up region finder")
        self.dataset_maker = SpectrumDatasetMaker(
            containment_correction=False, selection=["counts", "exposure", "edisp"]
        )

        self.region_finder = WobbleRegionsFinder(n_off_regions=n_off_regions)
        self.bkg_maker = ReflectedRegionsBackgroundMaker(
            region_finder=self.region_finder
        )

    def create_safe_mask_min_aeff(
        self,
        aeff_percent=10,
        bias_percent=10,
        position=None,
        fixed_offset=None,
        offset_max="3 deg",
    ):
        self.log.info("Create safe mask from minimum aeff max percentage")
        self.safe_mask_masker = SafeMaskMaker(
            methods=["aeff-default"],
            aeff_percent=aeff_percent if aeff_percent is not None else 10,
            bias_percent=10 if bias_percent is not None else 10,
            position=None,
            fixed_offset=None,
            offset_max=offset_max if offset_max is not None else "3 deg",
        )

    def add_source_to_exclusion_region(self, source_name=None, radius=0.1 * u.deg, reset=False):

        try:
            self.exclusion_regions
            assert reset == False
        except:
            self.exclusion_regions = []

        if source_name != None:
            if isinstance(source_name, str):
                exclusion_region = CircleSkyRegion(
                    center=SkyCoord.from_name("Crab Nebula", frame="galactic"),
                    radius=radius,
                )
            elif isinstance(source_name, SkyCoord):
                exclusion_region = CircleSkyRegion(
                    center=source_name.galactic,
                    radius=radius,
                )
            self.exclusion_regions.append(exclusion_region)

    def add_exclusion_region(self, region=None, reset=True):
        try:
            assert region != None
            self.exclusion_regions.append(region)
            assert reset == False
        except:
            self.exclusion_regions = []

    def create_exclusion_mask(self):
        skydir = self.src_pos.galactic
        excluded_geom = WcsGeom.create(
            npix=(125, 125), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
        )
        if len(self.exclusion_regions) == 0:
            self.log.info("Creating empty/dummy exclusion region")
            pos = SkyCoord(0, 90, unit="deg")
            exclusion_region = CircleSkyRegion(pos, 0.00001 * u.deg)
            self.exclusion_mask = ~excluded_geom.region_mask([exclusion_region])
        else:
            self.log.info("Creating exclusion region")
            self.exclusion_mask = ~excluded_geom.region_mask(self.exclusion_regions)

    def create_datasets(self):

        self.datasets = Datasets()
        self.counts_off_array = []

        for obs in self.observations_total:

            try:
                self.log.info("Run ID {}".format(obs.obs_id))

                central_pointing = SkyCoord(
                    obs.obs_info["RA_PNT"],
                    obs.obs_info["DEC_PNT"],
                    frame="icrs",
                    unit="deg",
                )

                self.region_finder.run(self.on_region, central_pointing)

                self.region_finder.run(self.on_region, central_pointing)

                if obs.fixed_pointing_info.meta["OBS_MODE"] == "UNDETERMINED":
                    self.log.warning("OBS_MODE for run {} is UNDETERMINED, wrong offset?".format(obs.obs_id))
                    obs.fixed_pointing_info.meta["OBS_MODE"] = "WOBBLE"

                dataset = self.dataset_maker.run(
                    self.dataset_template.copy(name=str(obs.obs_id)), obs
                )

                counts_off = self.bkg_maker.make_counts_off(dataset, obs)
                dataset_on_off = self.bkg_maker.run(dataset, obs)
                dataset_on_off.meta_table["SOURCE"] = self.target_name
                dataset_on_off = self.safe_mask_masker.run(dataset_on_off, obs)
                self.datasets.append(dataset_on_off)
                self.counts_off_array.append(counts_off)
            except IndexError:
                self.log.warning(
                    "Error processing run {}, skipping it".format(obs.obs_id)
                )

    def plot_pointings(self):
        # Check the OFF regions used for calculation of excess
        Fig = plt.figure(figsize=(8, 8))
        ax = self.exclusion_mask.plot()
        self.on_region.to_pixel(ax.wcs).plot(
            ax=ax, mfc="None", mew=np.random.random() + 1, marker="D"
        )
        plot_spectrum_datasets_off_regions(
            ax=ax, datasets=self.datasets, linewidth=2, legend=True
        )
        plt.grid()

        CS = ["C{}".format(k) for k in range(10)]
        # markers = ["*", "o", "+", "X", "s", "^", "v", "d"]

        for k, obs in enumerate(self.observations_total):
            point = PointSkyRegion(
                SkyCoord(
                    obs.obs_info["RA_PNT"],
                    obs.obs_info["DEC_PNT"],
                    frame="icrs",
                    unit="deg",
                )
            )
            point.on_region.meta = {"include": False}
            point.to_pixel(ax.wcs).plot(
                ax=ax,
                mfc="None",
                ms=np.random.random() * 30 + 10,
                mew=1.0,
                mec=CS[k % 10],
                marker="o",
            )

        return Fig

    def plot_excess_ts_livetime(self):
        info_table = self.datasets.info_table(cumulative=True)

        # Plot temporal evolution of excess events and significance value
        Fig = plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(
            np.sqrt(info_table["livetime"].to("h")),
            info_table["excess"],
            marker="o",
            ls="none",
        )
        plt.plot(info_table["livetime"].to("h")[-1:1], info_table["excess"][-1:1], "r")
        plt.xlabel("Sqrt Livetime h^(1/2)")
        plt.ylabel("Excess")
        plt.grid()
        plt.title("Excess vs Square root of Livetime")

        plt.subplot(122)
        plt.plot(
            np.sqrt(info_table["livetime"].to("h")),
            info_table["sqrt_ts"],
            marker="o",
            ls="none",
        )
        plt.grid()
        plt.xlabel("Sqrt Livetime h^(1/2)")
        plt.ylabel("sqrt_ts")
        plt.title("Significance vs Square root of Livetime")
        plt.subplots_adjust(wspace=0.5)
        return Fig

    def plot_counts_exposure_edisp_per_obs(self):
        plt.figure(figsize=(18, len(self.datasets) * 5))
        j = 1

        for data in self.datasets:
            plt.subplot(len(self.datasets), 3, j)
            data.plot_counts()
            data.plot_excess()
            plt.grid(which="both")
            plt.title(f"Run {data.name} Counts and Excess")
            j += 1

            plt.subplot(len(self.datasets), 3, j)
            data.exposure.plot()
            plt.grid(which="both")
            plt.title(f"Run {data.name} Exposure")
            j += 1

            plt.subplot(len(self.datasets), 3, j)
            if data.edisp is not None:
                kernel = data.edisp.get_edisp_kernel()
                kernel.plot_matrix(add_cbar=True)
                plt.title(f"Run {data.name} Energy Dispersion")
            j += 1
        plt.subplots_adjust(hspace=0.3)

    def get_pivot_energy(self):
        """
        Using Power Law spectral model with the given reference energy and
        get the decorrelation energy of the fit, within the fit energy range, e_edges
        """
        spectral_model = PowerLawSpectralModel(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )
        model = SkyModel(spectral_model=spectral_model, name=self.target_name)
        model_check = model.copy()

        # Stacked dataset method
        stacked_dataset = Datasets(self.datasets).stack_reduce()
        stacked_dataset.models = model_check

        fit_stacked = Fit()
        result_stacked = fit_stacked.run(datasets=stacked_dataset)

        return model_check.spectral_model.pivot_energy
