import json
import os


class JsonDump:
    def __init__(self, path, name):
        
        self._path = path
        self._data = {}
        self._out_name = f"{self._path}/{name}.json"
        
    def dump_json(self):

        self._create_out_dir()

        with open(self._out_name, 'w') as f:
            json.dump(self._data, f, indent=4)

    def collect_corruption_data(self, corruption: str, eval_matrix: str, value: float):
        if eval_matrix not in self._data:
            self._data[eval_matrix] = {}
        self._data[eval_matrix][corruption] = value
    
    def collect_general_data(self, eval_matrix: str, value: float):
        self._data[eval_matrix] = value

    def _create_out_dir(self):
        if not os.path.isdir(self._path):
            os.mkdir(self._path)
