import logging
import xmltodict

import astropy.units as u

from gammapy.maps import Map
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    SkyModel,
    PointSpatialModel,
    TemplateSpatialModel,
    PowerLawSpectralModel,
    LogParabolaSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    SuperExpCutoffPowerLaw4FGLDR3SpectralModel,
    PowerLawNormSpectralModel,
)


class FermiSkyModel(object):
    def __init__(self, xml_f, aux_path):
        self.xml_f = xml_f
        self.aux_path = aux_path
        self._set_logging()
        self.ebl_absorption = None

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def set_target_name(self, name):
        self.target_name = name

    def set_iso_diffuse(self, iso_diffuse):
        self.iso_diffuse = iso_diffuse
        self.iso_diffuse.spectral_model.model2.parameters[0].min = 0
        self.iso_diffuse.spectral_model.model2.parameters[0].max = 10

    def set_gal_diffuse(self, gal_diffuse):
        self.gal_diffuse = gal_diffuse

    def set_ebl_absorption(self, ebl_absorption):
        self.ebl_absorption = ebl_absorption

    @staticmethod
    def amplitude(var, name="amplitude", is_src_target=False):
        P = Parameter(name=name, value=0)
        P.value = float(var["@value"]) * float(var["@scale"]) * 1e6
        P.min = float(var["@min"]) * float(var["@scale"]) * 1e6
        P.max = float(var["@max"]) * float(var["@scale"]) * 1e6
        P.frozen = bool(var["@free"] == "0") and not is_src_target
        P.unit = u.Unit("1/(cm2*s*TeV)")
        P._is_norm = True
        return P

    @staticmethod
    def index(var, name="index", keep_sign=False, is_src_target=False):
        P = Parameter(name=name, value=0)
        if name == "index" and not keep_sign:
            Sign = -1
        else:
            Sign = +1
        P.value = Sign * float(var["@value"])
        P.min = Sign * float(var["@min"])
        P.max = Sign * float(var["@max"])
        P.frozen = bool(var["@free"] == "0") and not is_src_target
        return P

    @staticmethod
    def lambda_(var, name="lambda_", is_src_target=False):
        P = Parameter(name=name, value=0)
        P.value = 1.0 / float(var["@value"])
        P.min = 1.0 / float(var["@max"])
        P.max = 1.0 / float(var["@min"])
        P.unit = u.Unit("1/(MeV)")
        P.frozen = bool(var["@free"] == "0") and not is_src_target
        return P

    @staticmethod
    def reference(var, name="reference", is_src_target=False):
        P = Parameter(name=name, value=0)
        P.value = float(var["@value"]) * float(var["@scale"]) * 1e-6
        P.min = float(var["@min"]) * float(var["@scale"]) * 1e-6
        P.max = float(var["@max"]) * float(var["@scale"]) * 1e-6
        P.frozen = bool(var["@free"] == "0")  # and not is_src_target
        P.unit = u.TeV
        return P

    def powerlaw(self, spectrum, is_src_target=False):
        model = PowerLawSpectralModel()
        model.amplitude = self.amplitude(
            spectrum[0], name="amplitude", is_src_target=is_src_target
        )
        model.index = self.index(spectrum[1], name="index", is_src_target=is_src_target)
        model.reference = self.reference(
            spectrum[2], name="reference", is_src_target=is_src_target
        )
        return model

    def powerlaw_eblatten(self, spectrum, is_src_target=False):
        model = PowerLawSpectralModel()
        model.amplitude = self.amplitude(
            spectrum[0], name="amplitude", is_src_target=is_src_target
        )
        model.index = self.index(
            spectrum[1], name="index", keep_sign=True, is_src_target=is_src_target
        )
        model.reference = self.reference(
            spectrum[3], name="reference", is_src_target=is_src_target
        )
        return model

    def logparabola(self, spectrum, is_src_target=False):
        model = LogParabolaSpectralModel()
        model.amplitude = self.amplitude(
            spectrum[0], name="amplitude", is_src_target=is_src_target
        )
        model.alpha = self.index(spectrum[1], name="alpha", is_src_target=is_src_target)
        model.beta = self.index(spectrum[2], name="beta", is_src_target=is_src_target)
        model.reference = self.reference(
            spectrum[3], name="reference", is_src_target=is_src_target
        )
        return model

    def plexpcutoff(self, spectrum, is_src_target=False):
        model = ExpCutoffPowerLawSpectralModel()
        model.amplitude = self.amplitude(
            spectrum[0], name="amplitude", is_src_target=is_src_target
        )
        model.index = self.index(spectrum[1], name="index", is_src_target=is_src_target)
        model.reference = self.reference(
            spectrum[2], name="reference", is_src_target=is_src_target
        )
        model.lambda_ = self.lambda_(
            spectrum[3], name="lambda_", is_src_target=is_src_target
        )
        model.alpha = self.index(spectrum[4], name="alpha", is_src_target=is_src_target)
        return model

    def plexpcutoff4(self, spectrum, is_src_target=False):
        model = SuperExpCutoffPowerLaw4FGLDR3SpectralModel()
        model.amplitude = self.amplitude(
            spectrum[0], name="amplitude", is_src_target=is_src_target
        )
        model.index_1 = self.index(
            spectrum[1], name="index_1", is_src_target=is_src_target
        )
        model.reference = self.reference(
            spectrum[2], name="reference", is_src_target=is_src_target
        )
        model.expfactor = self.index(
            spectrum[3], name="expfactor", is_src_target=is_src_target
        )
        model.index_2 = self.index(
            spectrum[4], name="index_2", is_src_target=is_src_target
        )
        return model

    def create_iso_diffuse_skymodel(self):
        source = self.iso_diffuse
        # source.parameters["norm"].min = 0
        # source.parameters["norm"].max = 10
        # source.parameters["norm"].frozen = False
        self.log.info(" -> {0}".format("Iso diffuse"))
        return source

    def create_gal_diffuse_skymodel(self):
        template_diffuse = TemplateSpatialModel(self.gal_diffuse, normalize=False)
        source = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=template_diffuse,
            name="diffuse-iem",
        )
        source.parameters["norm"].min = 0
        source.parameters["norm"].max = 10
        source.parameters["norm"].frozen = False
        self.log.info(" -> {0}".format("Galactic diffuse"))
        return source

    def create_source_skymodel(self, src, lp_is_intrinsic=False):
        source_name = src["@name"]
        # srctype = src["@type"]
        spectrum_type = src["spectrum"]["@type"].split("EblAtten::")[-1]
        spectrum = src["spectrum"]["parameter"]
        # spatial_type = src["spatialModel"]["@type"]
        spatial_pars = src["spatialModel"]["parameter"]

        source_name_red = source_name.replace("_", "").replace(" ", "")
        target_red = self.target_name.replace("_", "").replace(" ", "")

        if source_name_red == target_red:
            source_name = self.target_name
            is_src_target = True
            self.log.debug("Detected target source")
        else:
            is_src_target = False

        if spectrum_type == "PowerLaw":
            model = self.powerlaw(spectrum, is_src_target)
        elif spectrum_type == "LogParabola" and "EblAtten" in src["spectrum"]["@type"]:
            if lp_is_intrinsic:
                model = self.logparabola(spectrum, is_src_target)
            else:
                model = self.powerlaw_eblatten(spectrum, is_src_target)
        elif spectrum_type == "LogParabola":
            model = self.logparabola(spectrum, is_src_target)
        elif spectrum_type == "PLSuperExpCutoff":
            model = self.plexpcutoff(spectrum, is_src_target)
        elif spectrum_type == "PLSuperExpCutoff4":
            model = self.plexpcutoff4(spectrum, is_src_target)
        else:
            print(spectrum_type)

        self.log.info(
            " -> {0}, {1}, frozen? {2}".format(
                source_name, spectrum_type, str(model.amplitude.frozen)
            )
        )

        if is_src_target and self.ebl_absorption != None:
            model = model * self.ebl_absorption

        if src["spatialModel"]["@type"] == "SkyDirFunction":
            spatial_model = PointSpatialModel(
                lon_0="{} deg".format(spatial_pars[0]["@value"]),
                lat_0="{} deg".format(spatial_pars[1]["@value"]),
                frame="fk5",
            )
        elif src["spatialModel"]["@type"] == "SpatialMap":
            file_name = src["spatialModel"]["@file"].split("/")[-1]
            file_path = f"{self.aux_path}/Templates/{file_name}"
            m = Map.read(file_path)
            m = m.copy(unit="sr^-1")
            spatial_model = TemplateSpatialModel(m, filename=file_path)

        spatial_model.freeze()
        source = SkyModel(
            spectral_model=model,
            spatial_model=spatial_model,
            name=source_name,
        )

        return source

    def create_full_skymodel(self, lp_is_intrinsic=False):

        self.list_sources = []

        with open(self.xml_f) as f:
            data = xmltodict.parse(f.read())["source_library"]["source"]
            self.list_of_sources_final = [src["@name"] for src in data]

        par_to_val = lambda par: float(par["@value"]) * float(par["@scale"])
        for k, src in enumerate(data):
            source_name = src["@name"]
            if source_name == "IsoDiffModel":
                source = self.create_iso_diffuse_skymodel()
            elif source_name == "GalDiffModel":
                source = self.create_gal_diffuse_skymodel()
            else:
                source = self.create_source_skymodel(src, lp_is_intrinsic)

            self.list_sources.append(source)
