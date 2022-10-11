import gzip
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
from gammapy.data import DataStore,ObservationFilter
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.irf import (
    EDispKernel,
    PSFMap,
    EDispMap,
    EDispKernelMap,
)
from gammapy.maps import Map,WcsGeom,MapAxis,RegionGeom
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
from regions import PointSkyRegion,CircleSkyRegion

#from .skymodel import FermiSkyModel
from .files import Files
from ..utils.plotting import new_viridis

from pathlib import Path
import logging


#from gammapy.estimators import FluxPointsEstimator,LightCurveEstimator
from gammapy.visualization import plot_spectrum_datasets_off_regions
from gammapy.analysis import Analysis, AnalysisConfig

from ..utils.datastore import make_obs_hdu_index

class InstrumentResponse(Files):
    pass 
    

class Observations(Files):
    def create_datastore(self,overwrite=False):
        self.log.info("Creating datastore from {}".format(self.dl3_path))
        try:
            if overwrite:
                raise OSError
            self.datastore = DataStore.from_dir(self.dl3_path)
        except OSError:
            self.log.warning("obs/hdu index files not found, creating them from {}".format(self.dl3_path))
            make_obs_hdu_index(self.dl3_path)
            self.datastore = DataStore.from_dir(self.dl3_path)
    def get_targetname(self):
        self.targetname = np.unique(self.datastore.obs_table["OBJECT"])[0]
        self.log.info("The source is {}".format(self.targetname))
    
    def set_targetname(self,name):
        self.targetname = name
        self.log.info("The source is {}".format(self.targetname))
    
    def get_observations(self):
        self.obs_list  = self.datastore.obs_table["OBS_ID"].data
        self.observations_total = self.datastore.get_observations(
            self.obs_list, 
            required_irf=["aeff", "edisp", "rad_max"], 
            skip_missing=False,
        )

class EnergyAxes(Files):
     def set_energy_axes(self,eedges=None,emin=None,emax=None,nbins=None):
        self.log.info("Setting energy axes") 
        
        if eedges == None:
            if emin == None:
                emin = 10**1 * u.GeV
            if emax == None:
                emax = 10**5 * u.GeV
            if nbins == None:
                nbins = int(np.log10(emax/emin)*6+0.5)
            eedger = np.logspace(np.log10(float(emin/u.GeV)),
                                 np.log10(float(emax/u.GeV)),
                                 int(nbins+1.5))*u.GeV
            eedget = np.logspace(np.log10(float(emin/u.GeV)),
                                 np.log10(float(emax/u.GeV)),
                                 int(nbins*1.4+1.5))*u.GeV
        
        self.energy_axis      = MapAxis.from_energy_edges(eedger)
        self.energy_axis_true = MapAxis.from_energy_edges(eedget).copy(name="energy_true")
        
class Analysis1D(Observations,EnergyAxes):
    def set_src_pos(self,src_pos=None,targetname=None):
        if src_pos != None:
            self.src_pos = src_pos
        else:
            if targetname == None:
                targetname = self.targetname
            self.src_pos = SkyCoord.from_name(targetname)
        self.log.info("Setting source position to {}".format(self.src_pos)) 
    
    def set_on_region(self): 
        self.log.info("Setting on region and empty dataset template")
        self.on_region = PointSkyRegion(self.src_pos)
        ### Hack to allow for Fermi+IACT fit (otherwise pointskyregion.contains returns None)
        self.on_region.meta = {'include': False}
        geom = RegionGeom.create(region=self.on_region, axes=[self.energy_axis])
        self.dataset_template = SpectrumDataset.create(
            geom=geom, energy_axis_true=self.energy_axis_true
        )
        
    def run_region_finder(self,n_off_regions=1):
        self.log.info("Setting up region finder")
        self.dataset_maker = SpectrumDatasetMaker(
            containment_correction=False, 
            selection=["counts", "exposure", "edisp"]
        )

        self.region_finder = WobbleRegionsFinder(n_off_regions=n_off_regions)
        self.bkg_maker = ReflectedRegionsBackgroundMaker(
            region_finder=self.region_finder
        )
        
    def create_safe_mask_min_aeff(self,aeff_percent=10,bias_percent=10,position=None,fixed_offset=None,offset_max="3 deg"):
        self.log.info("Create safe mask from minimum aeff max percentage")
        self.safe_mask_masker = SafeMaskMaker(methods=["aeff-default"], 
                                              aeff_percent=aeff_percent if aeff_percent is not None else 10,
                                              bias_percent=10 if bias_percent is not None else 10,
                                              position=None,
                                              fixed_offset=None,
                                              offset_max=offset_max if offset_max is not None else "3 deg",)
    
    def add_source_to_exclusion_region(self,src=None,radius=0.1*u.deg,reset=False):
        
        try:
            self.exclusion_regions
            assert(reset==False)
        except:
            self.exclusion_regions = []
        
        if src != None:
            if isinstance(src,str):
                exclusion_region = CircleSkyRegion(
                    center=SkyCoord.from_name("Crab Nebula", frame="galactic"),
                    radius=radius,
                )
            elif isinstance(src,SkyCoord):
                exclusion_region = CircleSkyRegion(
                    center=src.galactic,
                    radius=radius,
                )
            self.exclusion_regions.append(exclusion_region)
            
    def add_exclusion_region(self,region=None,reset=True):
        try:
            assert(region!=None)
            self.exclusion_regions.append(region)
            assert(reset==False)
        except:
            self.exclusion_regions = []
        
    
    def create_exclusion_mask(self):
        skydir = self.src_pos.galactic
        excluded_geom = WcsGeom.create(
            npix=(125, 125), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
        )
        if len(self.exclusion_regions) == 0:
            self.log.info("Creating empty/dummy exclusion region")
            pos = SkyCoord(0,90,unit='deg')
            exclusion_region = CircleSkyRegion(pos,0.00001*u.deg)
            self.exclusion_mask =~ excluded_geom.region_mask([exclusion_region])
        else:
            self.log.info("Creating exclusion region")
            self.exclusion_mask =~ excluded_geom.region_mask(self.exclusion_regions)
    
    def create_datasets(self):
        
        self.datasets = Datasets()
        self.counts_off_array = []
        
        for obs in self.observations_total:
            
            try:
                self.log.info("Run ID {}".format(obs.obs_id))
                    
                cen_pnt = SkyCoord(
                    obs.obs_info['RA_PNT'],
                    obs.obs_info['DEC_PNT'],
                    frame="icrs", unit="deg")

                self.region_finder.run(self.on_region,cen_pnt)
                
                dataset = self.dataset_maker.run(
                    self.dataset_template.copy(name=str(obs.obs_id)), obs
                )
                
                counts_off = self.bkg_maker.make_counts_off(dataset, obs)
                dataset_on_off = self.bkg_maker.run(dataset, obs)
                dataset_on_off.meta_table["SOURCE"]=self.targetname
                dataset_on_off = self.safe_mask_masker.run(dataset_on_off, obs)
                self.datasets.append(dataset_on_off)
                self.counts_off_array.append(counts_off)
            except IndexError:
                self.log.warning('Error processing run {}, skipping it'.format(obs.obs_id))
    
    def plot_pointings(self):
        # Check the OFF regions used for calculation of excess
        Fig = plt.figure(figsize=(8, 8))
        ax = self.exclusion_mask.plot()
        self.on_region.to_pixel(ax.wcs).plot(ax=ax,mfc='None',
                                             mew=np.random.random()+1, 
                                             marker='D')
        plot_spectrum_datasets_off_regions(ax=ax, 
                                           datasets=self.datasets, 
                                           linewidth=2, 
                                           legend=True)
        plt.grid()

        CS = ['C{}'.format(k) for k in range(10)]
        markers = ['*','o','+','X','s','^','v','d']

        for k,obs in enumerate(self.observations_total):
            point = PointSkyRegion(SkyCoord(
                obs.obs_info['RA_PNT'],
                obs.obs_info['DEC_PNT'],
                frame="icrs", unit="deg"))
            point.on_region.meta = {'include': False}
            point.to_pixel(ax.wcs).plot(ax=ax, mfc='None',
                                        ms=np.random.random()*30+10,
                                        mew=1.,
                                        mec=CS[k%10],
                                        marker='o')
            
        return(Fig)
    
    def plot_excess_ts_livetime(self):
        info_table = self.datasets.info_table(cumulative=True)

        # Plot temporal evolution of excess events and significance value
        Fig = plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.plot(
            np.sqrt(info_table["livetime"].to("h")), 
            info_table["excess"], 
            marker="o", 
            ls="none"
        )
        plt.plot(info_table["livetime"].to("h")[-1:1], 
                 info_table["excess"][-1:1], 
                 'r')
        plt.xlabel("Sqrt Livetime h^(1/2)")
        plt.ylabel("Excess")
        plt.grid()
        plt.title('Excess vs Square root of Livetime')

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
        plt.title('Significance vs Square root of Livetime')
        plt.subplots_adjust(wspace=0.5)
        return(Fig)

    def plot_counts_exposure_edisp_per_obs(self):
        plt.figure(figsize=(18, len(self.datasets)*5))
        j=1

        for data in self.datasets:
            plt.subplot(len(self.datasets), 3, j)
            data.plot_counts()
            data.plot_excess()
            plt.grid(which="both")
            plt.title(f'Run {data.name} Counts and Excess')
            j += 1
            
            plt.subplot(len(self.datasets), 3, j)
            data.exposure.plot()
            plt.grid(which='both')
            plt.title(f'Run {data.name} Exposure')
            j += 1
            
            plt.subplot(len(self.datasets), 3, j)
            if data.edisp is not None:
                kernel = data.edisp.get_edisp_kernel()
                kernel.plot_matrix(add_cbar=True)
                plt.title(f'Run {data.name} Energy Dispersion')
            j += 1
        plt.subplots_adjust(hspace=0.3)
        
    def get_pivot_energy(self):
        """
        Using Power Law spectral model with the given reference energy and 
        get the decorrelation energy of the fit, within the fit energy range, e_edges
        """
        spectral_model = PowerLawSpectralModel(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1*u.TeV
        )
        model = SkyModel(spectral_model=spectral_model, name=self.targetname)
        model_check = model.copy()

        # Stacked dataset method
        stacked_dataset = Datasets(self.datasets).stack_reduce()
        stacked_dataset.models = model_check

        fit_stacked = Fit()
        result_stacked = fit_stacked.run(datasets=stacked_dataset)

        return model_check.spectral_model.pivot_energy

    
    
    
