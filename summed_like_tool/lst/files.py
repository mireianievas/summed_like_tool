
import logging

class Files(object):
    def __init__(self,lst_path,srcmodel):
        self.lst_path  = lst_path
        self.model = srcmodel
        self._set_logging()
    
    def _set_logging(self):
        self.log = logging.getLogger(__name__)   
        self.log.setLevel(logging.INFO)  
    
    