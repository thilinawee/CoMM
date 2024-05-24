class TTADriver:
    def __init__(self): ...

    def read_params(self):
        """
        Read parameters from the user
        """
        ...

    def prepare_dataset(self, name: str, method: str):
        """
        Prepare the dataset according to the given distribution.
        """
        ...

    def prepare_data_loader(self, dataset):
        """
        Creates a data loader from a given dataset
        """
    def get_model(self):
        """
        Get the pretrained model
        """

    def prepare_data_loaders_for_sota_env(self):
        ...
    def apply_tta_for_sota_env(self):
        """
        Applies the test time adaptation algorithm to the original dataset with distribution shifts.
        """
        ...

    def test_for_sota_env(self):
        """
        Evaluate the adapted sota.
        """
        ...

    def prepare_data_loaders_for_novel_env(self):
        ...
    def apply_tta_for_novel_env(self):
        """
        Applies the test time adaptation algorithm to the label imbalanced dataset with distribution shifts.
        """
        ...

    def test_for_novel_env(self): ...

    def display_params(self):
        """
        Display the important parameters of the input script.
        """
        ...

    def collect_results_for_sota_env(self):
        """
        Collect the outcomes of the experiment.
        """
        ...

    def collect_results_for_novel_env(self): ...
