import os

class DirGen:
    def __init__(self, args):
        self.args = args
        self.root = self.args.report_dir

    def _dir_name(self):
        dataset = self.args.source_dataset.lower()
        batch_size = self.args.tta_batchsize
        n_drop_classes = len(self.args.drop_classes)
        severity = self.args.severity

        dir_name = f"{dataset}_bs{batch_size}_dc{n_drop_classes}_s{severity}"
        return dir_name
    
    def create_dir(self):
        
        dir_path = f"{self.root}/{self._dir_name()}"
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        return dir_path

