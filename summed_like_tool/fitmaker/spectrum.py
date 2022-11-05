import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from gammapy.modeling import Fit
from gammapy.datasets import Datasets, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator
from gammapy.maps import Map
from ..utils.various import closest_in_array


class FitMaker(object):
    def __init__(self, analyses, *args, **kwargs):
        self.set_analysis_objects(analyses)
        self.setup_fit(*args, **kwargs)

    def set_analysis_objects(self, analyses):
        self.analyses = analyses
        self.set_datasets([A.dataset for A in self.analyses])

    def set_datasets(self, datasets):
        self.datasets = Datasets()
        for k, d in enumerate(datasets):
            self.datasets.append(d)
            # if k==0: self.datasets.append(d)
            # else:    self.datasets.extend(d)

    def set_energy_mask(self, dataset, emin=100 * u.MeV, emax=30 * u.TeV):
        coords = dataset.counts.geom.get_coord()
        mask_energy = (coords["energy"] >= emin) * (coords["energy"] <= emax)
        dataset.mask_fit = Map.from_geom(geom=dataset.counts.geom, data=mask_energy)

    def setup_fit(self, *args, **kwargs):
        self.fit = Fit(*args, **kwargs)

    def global_fit(self, datasets=None):
        warnings.filterwarnings("ignore")
        if datasets == None:
            datasets = self.datasets
        self.result = self.fit.run(datasets=datasets)
        warnings.filterwarnings("default")

    def fit_energy_bin(self, energy_true, energy_reco, data):
        warnings.filterwarnings("ignore")
        for dataset in self.datasets:
            dataset.edisp_interp_kernel = EDispKernel(
                axes=[energy_true, energy_reco], data=data
            )

        fitBin = Fit(*args, **kwargs)
        result = self.fitBin.run(datasets=self.datasets)
        warnings.filterwarnings("default")
        return (self.fitBin, self.result)

    def set_target_source(self, targetname, dataset=None):
        if dataset != None:
            dataset = [dataset]
        else:
            dataset = self.datasets
        for d in dataset:
            for S in dataset.models:
                if S.name == targetname:
                    self.target_model = S
                    return None

    def print_fit_result(self):
        print(self.result)

    def print_parameters(self, first=True, full=False):

        if first:
            datasets = [
                self.datasets[0],
            ]
        else:
            datasets = self.datasets

        for d in datasets:
            if full:
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
        self.ebin_edges = (
            np.append(self.lat_ebin["col2"][0], self.lat_ebin["col3"]) * u.MeV
        )

    def get_spectral_points(self, ebin_edges=None, targetname=None, datasets=None):
        warnings.filterwarnings("ignore")

        if datasets == None:
            datasets = self.datasets

        if ebin_edges != None:
            self.ebin_edges = ebin_edges

        if targetname == None:
            targetname = self.analyses[0].targetname
            # self.set_target_source(self.analyses[0],dataset=self.analyses[0].dataset)
            # self.set_target_source(targetname)

        fpe = FluxPointsEstimator(
            energy_edges=self.ebin_edges,
            source=self.target_model.name,
            n_sigma_ul=2,
            selection_optional="all",
        )

        self.flux_points = fpe.run(datasets=datasets)
        warnings.filterwarnings("default")

    def prepare_energy_bins(self, ebins_edges=None):
        energy_true_axis = dataset.edisp_interp_kernel.axes[0]

        for k, ebin_lo in ebin_edges[0:-1]:
            ebin_hi = ebins_edges[k + 1]
            energy_true_slice = MapAxis.from_energy_edges(np.append(ebin_lo, ebin_hi))

            for dataset in self.datasets:
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

    def plot_spectrum(self, ax=None, kwargs_fp=None, kwargs_model=None):

        energy_range = Quantity([self.ebin_edges[0], self.ebin_edges[-1]])

        if kwargs_fp is None:
            kwargs_fp = {
                "color": "black",
                "mfc": "gray",
                "marker": "D",
                "label": "Flux points",
            }
        if kwargs_model is None:
            kwargs_model = {
                "color": "gray",
                "label": "Best fit model",
            }
        self.flux_points.plot(ax=ax, sed_type="e2dnde", **kwargs_fp)

        spec = self.target_model.spectral_model
        spec.evaluate_error(energy_range)
        spec.plot(ax=ax, energy_bounds=energy_range, sed_type="e2dnde", **kwargs_model)

        kwargs_model_err = kwargs_model.copy()
        kwargs_model_err.pop("label", None)
        spec.plot_error(
            ax=ax, energy_bounds=energy_range, sed_type="e2dnde", **kwargs_model_err
        )
        return ax

    def plot_spectrum_enrico(
        self, ax=None, kwargs_fp=None, kwargs_model=None, kwargs_model_err=None
    ):
        ymean = self.lat_bute["col2"]
        yerrs = self.lat_bute["col3"]

        yerrp = ymean + yerrs
        yerrn = ymean - yerrs  # 10**(2*np.log10(ymean)-np.log10(yerrp))

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
            kwargs_model_err["label"] = "enrico/fermitools"

        # Best-Fit model
        ax.plot(
            self.lat_bute["col1"] * u.MeV,
            ymean * u.Unit("erg/(cm2*s)"),
            **kwargs_model,
        )
        # confidence band
        ax.fill_between(
            x=self.lat_bute["col1"] * u.MeV,
            y1=yerrn * u.Unit("erg/(cm2*s)"),
            y2=yerrp * u.Unit("erg/(cm2*s)"),
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
            yerr=lat_ebin["col5"] * u.Unit("erg/(cm2*s)"),
            **kwargs_fp,
        )

        ax.set_ylim(
            [
                min(self.lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 0.2,
                max(self.lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 2,
            ]
        )

        return ax

    def plot_spectrum_fold(self, ax=None, foldfile=None, kwargs=None):
        import uproot

        if not os.path.exists(foldfile):
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

        fold = uproot.open(foldfile)
        sed = fold["observed_sed"].tojson()

        x = sed["fX"]
        y = sed["fY"]
        xerrlow = sed["fEXlow"]
        xerrhigh = sed["fEXhigh"]
        yerrlow = sed["fEYlow"]
        yerrhigh = sed["fEYhigh"]

        self.ax.errorbar(
            x=x * u.GeV,
            y=y * u.Unit("TeV/(cm2 * s)"),
            xerr=[xerrlow * u.GeV, xerrhigh * u.GeV],
            yerr=[
                yerrlow * u.Unit("TeV/(cm2 * s)"),
                yerrhigh * u.Unit("TeV/(cm2 * s)"),
            ],
            **kwargs,
        )

        return ax
