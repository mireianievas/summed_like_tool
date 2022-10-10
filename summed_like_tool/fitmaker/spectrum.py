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
    
    def __init__(self,analyses,*args,**kwargs):
        self.set_analysis_objects(analyses)
        self.setup_fit(*args,**kwargs)
    
    def set_analysis_objects(self,analyses):
        self.analyses = analyses
        self.set_datasets([A.dataset for A in self.analyses])
    
    def set_datasets(self,datasets):
        self.datasets = Datasets()
        for k,d in enumerate(datasets):
            self.datasets.append(d)
            #if k==0: self.datasets.append(d)
            #else:    self.datasets.extend(d)
    
    def set_energy_mask(self,dataset,emin=100*u.MeV,emax=30*u.TeV):
        coords = dataset.counts.geom.get_coord()
        mask_energy = (coords["energy"] >= emin)*(coords["energy"] <= emax)
        dataset.mask_fit = Map.from_geom(
            geom=dataset.counts.geom, data=mask_energy
        )
    
    def setup_fit(self,*args,**kwargs):
        self.fit = Fit(*args,**kwargs)
    
    def global_fit(self):
        warnings.filterwarnings("ignore")
        self.result = self.fit.run(datasets=self.datasets)
        warnings.filterwarnings("default")
    
    def fit_energy_bin(self,energy_true,energy_reco,data):
        warnings.filterwarnings("ignore")
        for dataset in self.datasets:
            dataset.edisp_interp_kernel = EDispKernel(axes=[energy_true,energy_reco],
                                                      data=data)
        
        fitBin = Fit(*args,**kwargs)
        result = self.fitBin.run(datasets=self.datasets)
        warnings.filterwarnings("default")
        return(self.fitBin,self.result)
    
    def set_target_source(self,targetname,dataset=None):
        if dataset == None:
            dataset = self.datasets[0]
        
        for S in dataset.models:
            if S.name == targetname:
                self.target_model = S
    
    def print_fit_result(self):
        print(self.result)
    
    def print_parameters(self,first=True,full=False):
        
        if first:
            datasets = [self.datasets[0],]
        else:
            datasets = self.datasets
        
        for d in datasets:
            if full:
                d.models.to_parameters_table().pprint_all()
            else:
                d.models.to_parameters_table().pprint()
        

class SpectralAnalysis(FitMaker):
    
    def read_enrico_spectrum(self,lat_ebin_file=None,lat_bute_file=None):
        if lat_ebin_file == None:
            lat_ebin_file = self.analyses[0].lat_ebin_file[0]
        if lat_bute_file == None:
            lat_bute_file = self.analyses[0].lat_bute_file[0]
            
        self.lat_ebin   = Table.read(lat_ebin_file,format='ascii')
        self.lat_bute   = Table.read(lat_bute_file,format='ascii')
        self.ebin_edges = np.append(self.lat_ebin['col2'][0],
                                    self.lat_ebin['col3'])*u.MeV
        
    def get_spectral_points(self,ebin_edges=None,targetname=None,datasets=None):
        warnings.filterwarnings("ignore")
        
        if datasets==None:
            datasets = self.datasets
        
        if ebin_edges != None:
            self.ebin_edges = ebin_edges
            
        if targetname != None:
            self.target_model = self.get_target_source(targetname)
        
        fpe = FluxPointsEstimator(energy_edges=self.ebin_edges, 
                                source=self.target_model.name,
                                n_sigma_ul=2,
                                selection_optional='all')
        self.flux_points = fpe.run(datasets=datasets)
        warnings.filterwarnings("default")
        
    def prepare_energy_bins(self,ebins_edges=None):
        energy_true_axis = dataset.edisp_interp_kernel.axes[0]
        
        for k,ebin_lo in ebin_edges[0:-1]:
            ebin_hi = ebins_edges[k+1]
            energy_true_slice = MapAxis.from_energy_edges(np.append(ebin_lo,ebin_hi))
            
            for dataset in self.datasets:
                energy_reco_axis_slice,jmin,jmax = slice_in_mapaxis(energy_reco_axis,ebin_lo,ebin_hi,2)
                #energy_true_axis_slice,imin,imax = slice_in_mapaxis(energy_true_axis,ebin_lo,ebin_hi,0)
                
                drm_interp = dataset.edisp_interp_kernel.valuate(
                **{'energy':energy_reco_axis,'energy_true':energy_true_slice})
            
                dataset.edisp_interp_kernel = EDispKernel(axes=[axis_true,axis_reco],
                                          data=np.asarray(drm_interp))
                
                
            self.fit_energy_bin()
        
    
    def plot_spectrum(self):
        self.Fig = plt.figure(dpi=150)

        energy_range = Quantity([self.ebin_edges[0], self.ebin_edges[-1]])
        
        try:
            self.ax = self.flux_points.plot(
                sed_type="e2dnde",color="black",mfc='gray',marker='D'
            )
        except: # AttributeError
            self.ax = self.Fig.add_subplot(111)
        
        spec = self.target_model.spectral_model
        spec.evaluate_error(energy_range)
        spec.plot(energy_bounds=energy_range, sed_type="e2dnde",ax=self.ax,color='gray')
        spec.plot_error(energy_bounds=energy_range,sed_type="e2dnde",ax=self.ax,label='gammapy')

        #ax.set_ylim([4e-12,3e-10])
        #ax.set_xlim([2e2,1.2e6])

        self.ax.set_title(self.target_model.name)
        self.ax.legend()

    
    def plot_spectrum_enrico(self):
        ymean = self.lat_bute['col2']
        yerrs = self.lat_bute['col3']

        yerrp = ymean+yerrs
        yerrn = ymean-yerrs #10**(2*np.log10(ymean)-np.log10(yerrp))

        self.ax.plot(
            self.lat_bute['col1']*u.MeV,
            ymean*u.Unit("erg/(cm2*s)"),
            color='red',
            zorder=-10,
        )
        self.ax.fill_between(
            x=self.lat_bute['col1']*u.MeV,
            y1=yerrn*u.Unit("erg/(cm2*s)"),
            y2=yerrp*u.Unit("erg/(cm2*s)"),
            color='red',
            alpha=0.2,
            zorder=-10,
            label='enrico/fermitools',
        )

        lat_ebin = Table(self.lat_ebin)
        isuplim = lat_ebin['col5']==0
        lat_ebin['col5'][isuplim] = lat_ebin['col4'][isuplim]*0.5

        self.ax.errorbar(
            x = lat_ebin['col1']*u.MeV,
            y = lat_ebin['col4']*u.Unit("erg/(cm2*s)"),
            xerr = [lat_ebin['col1']*u.MeV-lat_ebin['col2']*u.MeV,
                    lat_ebin['col3']*u.MeV-lat_ebin['col1']*u.MeV],
            yerr = lat_ebin['col5']*u.Unit("erg/(cm2*s)"),
            marker='o',
            ls='None',
            color='red',
            mfc='white',
            zorder=-10,
            uplims=isuplim
        )

        self.ax.set_ylim([min(self.lat_ebin['col4']*u.Unit("erg/(cm2*s)"))*0.2,
                     max(self.lat_ebin['col4']*u.Unit("erg/(cm2*s)"))*2])
        self.ax.set_xlim([self.ebin_edges[0]*0.7,self.ebin_edges[-1]*1.4])

        self.ax.set_title(self.target_model.name)
        self.ax.legend()
        return(self.Fig)
    
    def plot_spectrum_fold(self,foldfile=None):
        import uproot
        if not os.path.exists(foldfile):
            return None
        
        fold = uproot.open(foldfile)
        sed = fold['observed_sed'].tojson()
        
        x = sed['fX']
        y = sed['fY']
        xerrlow  = sed['fEXlow']
        xerrhigh = sed['fEXhigh']
        yerrlow  = sed['fEYlow']
        yerrhigh = sed['fEYhigh']
        
        self.ax.errorbar(
            x = x*u.GeV,
            y = y*u.Unit("TeV/(cm2 * s)"),
            xerr = [xerrlow*u.GeV,
                    xerrhigh*u.GeV],
            yerr = [yerrlow*u.Unit("TeV/(cm2 * s)"),
                    yerrhigh*u.Unit("TeV/(cm2 * s)")],
            marker='o',
            ls='None',
            color='C4',
            mfc='white',
            zorder=-10,
            label='MAGIC/Fold',
            #uplims=isuplim
        )
        
        self.ax.legend()
        
        return(self.Fig)
    
