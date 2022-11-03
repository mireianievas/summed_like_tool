from summed_like_tool.fermi.files import Files

import logging
import xmltodict
from gammapy.modeling import Parameter

import astropy.units as u

from gammapy.modeling.models import (
    PowerLawSpectralModel,
    LogParabolaSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    SuperExpCutoffPowerLaw4FGLDR3SpectralModel,
    PointSpatialModel,
    SkyModel,
    TemplateSpatialModel,
    PowerLawNormSpectralModel,
)


class FermiSkyModel(object):
    def __init__(self, xml_f):
        self.xml_f = xml_f
        self._set_logging()
        self.ebl_absorption = None

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def set_target_name(self, name):
        self.targetname = name

    def set_isodiffuse(self, isodiffuse):
        self.isodiffuse = isodiffuse
        self.isodiffuse.spectral_model.model1.parameters[0].min = 0.001
        self.isodiffuse.spectral_model.model1.parameters[0].max = 10
        self.isodiffuse.spectral_model.model2.parameters[0].min = 0
        self.isodiffuse.spectral_model.model2.parameters[0].max = 10

    def set_galdiffuse(self, galdiffuse):
        self.galdiffuse = galdiffuse

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
    def index(var, name="index", keepsign=False, is_src_target=False):
        P = Parameter(name=name, value=0)
        if name == "index" and not keepsign:
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
            spectrum[1], name="index", keepsign=True, is_src_target=is_src_target
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

    def create_isodiffuse_skymodel(self):
        source = self.isodiffuse
        # source.parameters["norm"].min = 0
        # source.parameters["norm"].max = 10
        # source.parameters["norm"].frozen = False
        self.log.info(" -> {0}".format("Iso diffuse"))
        return source

    def create_galdiffuse_skymodel(self):
        template_diffuse = TemplateSpatialModel(self.galdiffuse, normalize=False)
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

    def create_point_source_skymodel(self, src):
        srcname = src["@name"]
        # srctype  = src["@type"]
        spectype = src["spectrum"]["@type"].split("EblAtten::")[-1]
        spectrum = src["spectrum"]["parameter"]
        # spatialtype = src["spatialModel"]["@type"]
        spatialpars = src["spatialModel"]["parameter"]

        srcname_red = srcname.replace("_", "").replace(" ", "")
        target_red = self.targetname.replace("_", "").replace(" ", "")

        if srcname_red == target_red:
            srcname = self.targetname
            is_src_target = True
            self.log.debug("Detected target source")
        else:
            is_src_target = False

        if spectype == "PowerLaw":
            model = self.powerlaw(spectrum, is_src_target)
        elif spectype == "LogParabola" and "EblAtten" in src["spectrum"]["@type"]:
            model = self.powerlaw_eblatten(spectrum, is_src_target)
        elif spectype == "LogParabola":
            model = self.logparabola(spectrum, is_src_target)
        elif spectype == "PLSuperExpCutoff":
            model = self.plexpcutoff(spectrum, is_src_target)
        elif spectype == "PLSuperExpCutoff4":
            model = self.plexpcutoff4(spectrum, is_src_target)
        else:
            print(spectype)

        self.log.info(
            " -> {0}, {1}, frozen? {2}".format(
                srcname, spectype, str(model.amplitude.frozen)
            )
        )

        if is_src_target and self.ebl_absorption != None:
            model = model * self.ebl_absorption

        spatial_model = PointSpatialModel(
            lon_0="{} deg".format(spatialpars[0]["@value"]),
            lat_0="{} deg".format(spatialpars[1]["@value"]),
            frame="fk5",
        )

        spatial_model.freeze()

        source = SkyModel(
            spectral_model=model,
            spatial_model=spatial_model,
            name=srcname,
        )
        return source

    def create_full_skymodel(self):

        self.list_sources = []

        with open(self.xml_f) as f:
            data = xmltodict.parse(f.read())["source_library"]["source"]
            self.list_of_sources_final = [src["@name"] for src in data]

        par_to_val = lambda par: float(par["@value"]) * float(par["@scale"])
        for k, src in enumerate(data):
            srcname = src["@name"]
            if srcname == "IsoDiffModel":
                source = self.create_isodiffuse_skymodel()
            elif srcname == "GalDiffModel":
                source = self.create_galdiffuse_skymodel()
            else:
                source = self.create_point_source_skymodel(src)

            self.list_sources.append(source)
