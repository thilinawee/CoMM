import logging


class TTALogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)

        self._create_console_handler()

    def _create_console_handler(self):
        formatter = logging.Formatter("%(levelname)s - %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.addHandler(ch)
