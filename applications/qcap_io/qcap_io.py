import os
from holoscan.core import Application
from holohub.qcap_source import QCAPSourceOp

class QcapIO(Application):
    def __init__(self):
        super().__init__()
        self.name = "Qcap IO App"

    def compose(self):
        pass
        
if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "qcap_io.yaml")
    app = QcapIO()
    app.config(config_file)
    app.run()
