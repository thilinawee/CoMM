
from logger.logger import TTALogger

logger = TTALogger(__file__)

class TTAConfig:
    # A singleton class to store the configuration of the TTA input scripts
    _instance = None
    _args = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    @classmethod
    def set_args(cls, args):
        cls._args = args

    @classmethod
    def get_args(cls):
        return cls._args
    
    def print_args(self):
        for key, value in vars(self._args).items():
            logger.info(f"{key}: {value}")

    


    
        