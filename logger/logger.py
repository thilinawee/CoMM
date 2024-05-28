import logging


class TTALogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)

        self._create_console_handler()

    def _create_console_handler(self):
        formatter = logging.Formatter("%(levelname)s - %(message)s")

        info_handler = logging.StreamHandler()
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        self.addHandler(info_handler)

        # debug_handler = logging.StreamHandler()
        # debug_handler.setLevel(logging.DEBUG)
        # debug_handler.setFormatter(formatter)
        # self.addHandler(debug_handler)
