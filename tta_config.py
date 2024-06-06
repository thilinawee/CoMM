import json

class TTAConfig:
    # A singleton class to store the configuration of the TTA input scripts
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._args = None
        
    def set_args(self, args):
        self._args = args

    def get_args(self):
        return self._args
    


    
        