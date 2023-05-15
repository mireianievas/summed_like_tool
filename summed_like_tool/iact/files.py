import logging
from os.path import dirname

class Files(object):
    def __init__(self, dl3_path, source_model):
        self.dl3_path = dl3_path
        self.dl3_dir  = dirname(dl3_path)
        self.model = source_model
        self._set_logging()

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
