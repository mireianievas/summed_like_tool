import os
import warnings
import numpy as np

import astropy.units as u
from astropy.table import Table
from astropy.units import Quantity

from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.irf import EDispKernel
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel

from ..iact.analysis import Analysis1D
from ..fermi.analysis import FermiAnalysis
from ..utils.various import slice_in_mapaxis #, closest_in_array


class FitMaker(object):
    def __init__(self, analyses, *args, **kwargs):
        self.set_analysis_objects(analyses)
        self.setup_fit(*args, **kwargs)

    def set_analysis_objects(self, analyses):
        self.analyses = analyses
        dsets = []
        for A in self.analyses:
            dsets.append(A.dataset)
        self.set_datasets(dsets)

    def set_datasets(self, datasets):
        self.datasets = Datasets()
        for k, d in enumerate(datasets):
            self.datasets.append(d)
            # if k==0: self.datasets.append(d)
            # else:    self.datasets.extend(d)

    def set_energy_mask(self, dataset, en_min=100 * u.MeV, en_max=30 * u.TeV):
        coords = dataset.counts.geom.get_coord()
        mask_energy = (coords["energy"] >= en_min) * (coords["energy"] <= en_max)
        dataset.mask_fit = Map.from_geom(geom=dataset.counts.geom, data=mask_energy)

    def setup_fit(self, *args, **kwargs):
        self.fit = Fit(*args, **kwargs)

    def global_fit(self, datasets=None):
        warnings.filterwarnings("ignore")
        if datasets == None:
            datasets = self.datasets
        self.result = self.fit.run(datasets=datasets)
        warnings.filterwarnings("default")

    def plot_parameter_stat_profile(self, datasets, parameter, ax=None):
        total_stat = self.result.total_stat

        parameter.scan_n_values = 20

        profile = self.fit.stat_profile(datasets=datasets, parameter=parameter)

        ax.plot(
            profile[f"{parameter.name}_scan"],
            profile["stat_scan"] - total_stat
        )
        ax.set_xlabel(f"{parameter.unit}")
        ax.set_ylabel("Delta TS")
        ax.set_title(
            f"{parameter.name}: {parameter.value:.2e} +/- {parameter.error:.2e}" +
            f"\n{parameter.value:.2e}+/- {parameter.error/parameter.value:.2f}"
        )

        return ax

    def fit_energy_bin(self, energy_true, energy_reco, data):
        warnings.filterwarnings("ignore")
        for dataset in self.datasets:
            dataset.edisp_interp_kernel = EDispKernel(
                axes=[energy_true, energy_reco], data=data
            )

        self.fitBin = Fit(*args, **kwargs)
        self.result = self.fitBin.run(datasets=self.datasets)
        warnings.filterwarnings("default")
        return (self.fitBin, self.result)

    def set_target_source(self, target_name, dataset=None):
        if dataset != None:
            dataset = [dataset]
        else:
            dataset = self.datasets
        for d in dataset:
            for S in dataset.models:
                if S.name == target_name:
                    self.target_model = S
                    return None

    def print_fit_result(self):
        print(self.result)

    def print_parameters(self, only_first_dataset=True, full_datasets=False):

        if only_first_dataset:
            datasets = [
                self.datasets[0],
            ]
        else:
            datasets = self.datasets

        for d in datasets:
            if full_datasets:
                d.models.to_parameters_table().pprint_all()
            else:
                d.models.to_parameters_table().pprint()


class SpectralAnalysis(FitMaker):
    def read_enrico_spectrum(self, lat_ebin_file=None, lat_bute_file=None):
        if lat_ebin_file == None:
            lat_ebin_file = self.analyses[0].lat_ebin_file[0]
        if lat_bute_file == None:
            lat_bute_file = self.analyses[0].lat_bute_file[0]

        self.lat_ebin = Table.read(lat_ebin_file, format="ascii")
        self.lat_bute = Table.read(lat_bute_file, format="ascii")
        self.energy_bin_edges = (
            np.append(self.lat_ebin["col2"][0], self.lat_ebin["col3"]) * u.MeV
        )

    def get_spectral_points(self, energy_bin_edges=None, target_name=None, datasets=None):
        warnings.filterwarnings("ignore")

        if datasets == None:
            datasets = self.datasets

        if energy_bin_edges != None:
            self.energy_bin_edges = energy_bin_edges

        if target_name == None:
            target_name = self.analyses[0].target_name
            # self.set_target_source(self.analyses[0],dataset=self.analyses[0].dataset)
            # self.set_target_source(target_name)

        fpe = FluxPointsEstimator(
            energy_edges=self.energy_bin_edges,
            source=self.target_model.name,
            n_sigma_ul=2,
            norm_min=1e-6, 
            norm_max=1e6,
            selection_optional="all",
        )

        self.flux_points = fpe.run(datasets=datasets)
        warnings.filterwarnings("default")

    def prepare_energy_bins(self, dataset, energy_bin_edges=None):
        energy_true_axis = dataset.edisp_interp_kernel.axes[0]

        for k, ebin_lo in energy_bin_edges[0:-1]:
            ebin_hi = energy_bin_edges[k + 1]
            energy_true_slice = MapAxis.from_energy_edges(
                np.append(ebin_lo, ebin_hi)
            )

            for dataset in self.datasets:
                ## Explanation/Description required
                energy_reco_axis_slice, jmin, jmax = slice_in_mapaxis(
                    energy_reco_axis, ebin_lo, ebin_hi, 2
                )
                # energy_true_axis_slice,imin,imax = slice_in_mapaxis(energy_true_axis,ebin_lo,ebin_hi,0)

                drm_interp = dataset.edisp_interp_kernel.valuate(
                    **{"energy": energy_reco_axis, "energy_true": energy_true_slice}
                )

                dataset.edisp_interp_kernel = EDispKernel(
                    axes=[axis_true, axis_reco], data=np.asarray(drm_interp)
                )

            self.fit_energy_bin()

    def plot_spectrum_fp(self, ax=None, flux_points=None, kwargs_fp=None):

        energy_range = Quantity([self.energy_bin_edges[0], self.energy_bin_edges[-1]])

        if kwargs_fp is None:
            kwargs_fp = {
                "sed_type": "e2dnde",
                "color": "black",
                "mfc": "gray",
                "marker": "D",
                "label": "Flux points",
            }
            
        if flux_points == None:
            flux_points = self.flux_points
        flux_points.plot(ax=ax, **kwargs_fp)
        return ax

    def plot_spectrum_model(self, ax=None, is_intrinsic=False, spec=None, kwargs_model=None):

        energy_range = Quantity([self.energy_bin_edges[0], self.energy_bin_edges[-1]])

        if kwargs_model is None:
            kwargs_model = {
                "sed_type": "e2dnde",
                "color": "gray",
            }

        if is_intrinsic:
            if spec==None:
                spec = self.target_model.spectral_model.model1
            if "label" not in kwargs_model.keys():
                kwargs_model["label"] = "Best fit intrinsic model - EBL deabsorbed"
            spec.plot(ax=ax, energy_bounds=energy_range, **kwargs_model)
        else:
            if spec==None:
                spec = self.target_model.spectral_model
            spec.evaluate_error(energy_range)

            kwargs_model_err = kwargs_model.copy()
            kwargs_model_err.pop("label", None)

            ax = spec.plot_error(
                ax=ax, energy_bounds=energy_range, **kwargs_model_err
            )
            
            if "facecolor" in kwargs_model:
                kwargs_model.pop("facecolor", None)
            
            ax = spec.plot(ax=ax, energy_bounds=energy_range, **kwargs_model)

        return ax

    def plot_residuals(self, ax=None, method="diff", kwargs_res=None):

        self.flux_points_dataset = FluxPointsDataset(
            data=self.flux_points, models=SkyModel(spectral_model=self.target_model.spectral_model)
        )

        self.flux_points_dataset.plot_residuals(ax=ax, method=method, **kwargs_res)

        return ax

    def plot_ts_profiles(self, ax=None, add_cbar=True, kwargs_ts=None):

        if kwargs_ts is None:
            kwargs_ts = {
                "sed_type": "e2dnde",
                "color": "darkorange",
            }
        self.flux_points.plot_ts_profiles(ax=ax, add_cbar=add_cbar, **kwargs_ts)
        ## add x_lim for energy bounds?

        return ax

    def plot_model_covariance_correlation(self, ax=None):

        spec = self.target_model.spectral_model
        spec.covariance.plot_correlation(ax=ax)

        return ax

    def plot_spectrum_enrico(
        self, ax=None, kwargs_fp=None, kwargs_model=None, kwargs_model_err=None
    ):
        y_mean = self.lat_bute["col2"]
        y_errs = self.lat_bute["col3"]

        y_errp = y_mean + y_errs
        y_errn = y_mean - y_errs  # 10**(2*np.log10(y_mean)-np.log10(y_errp))

        if ax == None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        if kwargs_fp is None:
            kwargs_fp = {
                "marker": "o",
                "ls": "None",
                "color": "red",
                "mfc": "white",
                "lw": 0.75,
                "mew": 0.75,
                "zorder": -10,
            }

        if kwargs_model is None:
            kwargs_model = {
                "color": "red",
                "lw": 0.75,
                "mew": 0.75,
                "zorder": -10,
            }

        if kwargs_model_err is None:
            kwargs_model_err = kwargs_model.copy()
            kwargs_model_err.pop("mew", None)
            kwargs_model_err["alpha"] = 0.2
            kwargs_model_err["label"] = "Enrico/Fermitools"

        # Best-Fit model
        ax.plot(
            self.lat_bute["col1"] * u.MeV,
            y_mean * u.Unit("erg/(cm2*s)"),
            **kwargs_model,
        )
        # confidence band
        ax.fill_between(
            x=self.lat_bute["col1"] * u.MeV,
            y1=y_errn * u.Unit("erg/(cm2*s)"),
            y2=y_errp * u.Unit("erg/(cm2*s)"),
            **kwargs_model_err,
        )

        # spectral points
        lat_ebin = Table(self.lat_ebin)
        isuplim = lat_ebin["col5"] == 0
        lat_ebin["col5"][isuplim] = lat_ebin["col4"][isuplim] * 0.5
        kwargs_fp["uplims"] = isuplim

        ax.errorbar(
            x=lat_ebin["col1"] * u.MeV,
            y=lat_ebin["col4"] * u.Unit("erg/(cm2*s)"),
            xerr=[
                lat_ebin["col1"] * u.MeV - lat_ebin["col2"] * u.MeV,
                lat_ebin["col3"] * u.MeV - lat_ebin["col1"] * u.MeV,
            ],
            yerr=np.abs(np.asarray(lat_ebin["col5"])) * u.Unit("erg/(cm2*s)"),
            **kwargs_fp,
        )

        ax.set_ylim(
            [
                min(self.lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 0.2,
                max(self.lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 2,
            ]
        )

        return ax

    def plot_spectrum_fold(self, ax=None, fold_file=None, kwargs=None):
        import uproot

        if not os.path.exists(fold_file):
            print(fold_file)
            return None

        if kwargs is None:
            kwargs = {
                "marker": "o",
                "ls": "None",
                "color": "C4",
                "mfc": "white",
                "zorder": -10,
                "label": "MAGIC/Fold",
                "lw": 0.75,
                "mew": 0.75,
                "alpha": 1,
            }

        fold = uproot.open(fold_file)
        sed = fold["observed_sed"].tojson()

        x = sed["fX"]
        y = sed["fY"]
        x_err_low = sed["fEXlow"]
        x_err_high = sed["fEXhigh"]
        y_err_low = sed["fEYlow"]
        y_err_high = sed["fEYhigh"]

        ax.errorbar(
            x=x * u.GeV,
            y=y * u.Unit("TeV/(cm2 * s)"),
            xerr=[x_err_low * u.GeV, x_err_high * u.GeV],
            yerr=[
                y_err_low * u.Unit("TeV/(cm2 * s)"),
                y_err_high * u.Unit("TeV/(cm2 * s)"),
            ],
            **kwargs,
        )

        return ax
