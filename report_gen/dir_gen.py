import os

class DirGen:
    def __init__(self, args):
        self.args = args
        self.root = self.args.report_dir

    def _dir_name(self):
        dataset = self.args.source_dataset.lower().replace("-", "_")
        batch_size = self.args.tta_batchsize
        n_drop_classes = len(self.args.drop_classes)
        severity = self.args.severity
        seed = self.args.seed

        dir_name = f"{dataset}/bs{batch_size}_dc{n_drop_classes}_s{severity}/seed_{seed}"
        return dir_name
    
    # def _create_root_dir(self):
    #     if not os.path.isdir(self.root):
    #         os.mkdir(self.root)

    def create_dir(self):
        # self._create_root_dir()
        
        dir_path = f"{self.root}/{self._dir_name()}"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        return dir_path

