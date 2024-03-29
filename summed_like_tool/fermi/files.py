from glob import glob
import logging


class Files(object):
    def __init__(self, lat_path, aux_path, source_model):
        self.path = lat_path
        self.aux_path = aux_path
        self.model = source_model
        self._set_logging()

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def discover_files(self):
        self.events_files = glob(f"{self.path}/*_MkTime.fits*")
        self.edrm_files = glob(f"{self.path}/*{self.model}_*eDRM.fits*")
        # xml_files    = glob(f"{self.path}/*{self.model}_model*.xml")
        # xml_final    = glob(f"{self.path}/*{self.model}*_out.xml")[0]
        self.xml_files = glob(f"{self.path}/*_out.xml")
        self.expmap_files = glob(f"{self.path}/*_BinnedMap.fits*")
        self.psf_files = glob(f"{self.path}/*_psf.fits*")
        self.diff_gal_files = glob(f"{self.aux_path}/gll_iem_v07.fits*")
        self.iso_files = glob(f"{self.aux_path}/iso_P8R3_SOURCE_V3_*.txt")

    def discover_spectra_result(self):
        self.lat_spectra = glob(f"{self.path}/Spectrum/SED*{self.model}*.dat")
        self.lat_bute_file = [
            K
            for K in self.lat_spectra
            if "cov" not in K
            and "Ebin" not in K
            and "ResData" not in K
            and "fitpars" not in K
        ]
        self.lat_ebin_file = [
            K for K in self.lat_spectra if "cov" not in K and "Ebin" in K
        ]

    def print_files(self):
        self.log.info("## Event files")
        self.log.info(self.events_files)
        self.log.info("## DRM files")
        self.log.info(self.edrm_files)
        self.log.info("## XML files")
        self.log.info(self.xml_files)
        self.log.info("## ExpMap files")
        self.log.info(self.expmap_files)
        self.log.info("## PSF files")
        self.log.info(self.psf_files)
        self.log.info("## Spectrum file")
        self.log.info(self.lat_bute_file)
        self.log.info("## Spectrum points")
        self.log.info(self.lat_ebin_file)

    def select_unique_files(self, key):
        self.unique_name = key
        var_list = [
            "events_files",
            "edrm_files",
            "expmap_files",
            "psf_files",
            "iso_files",
            #'diff_gal_files'
        ]
        for _v in var_list:
            if _v=='iso_files' and key in ['FRONTBACK', '']:
                filtered = [K for K in getattr(self, _v) if 'PSF' not in K and 'EDISP' not in K and 'FRONT' not in K and 'BACK' not in K]
                assert len(filtered) == 1
                setattr(self, _v.replace("_files", "_f"), filtered[0])
                continue
            try:
                filtered = [K for K in getattr(self, _v) if key in K]
                assert len(filtered) == 1
            except:
                print(
                    f"Variable self.{_v} does not contain one element after filtering by {key}"
                )
                print(filtered)
                raise
            else:
                setattr(self, _v.replace("_files", "_f"), filtered[0])

        self.xml_f = [f for f in self.xml_files if self.model in f][0]
        self.diff_gal_f = self.diff_gal_files[0]

    def print_selected_files(self):
        var_list = [
            "events_f",
            "edrm_f",
            "expmap_f",
            "psf_f",
            "iso_f",
            "diff_gal_f",
            "xml_f",
        ]

        for _v in var_list:
            self.log.info(getattr(self, _v))

    def prepare_files(self, key):
        self.tag = key
        self.discover_files()
        self.discover_spectra_result()
        self.select_unique_files(key)
