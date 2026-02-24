import sys
import os
from PySide6.QtWidgets import QApplication
from gui import RealSenseGUI
from experiment_config import load_recording_params

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'recording_config.json')
recording_params = load_recording_params(
    CONFIG_PATH,
    defaults={'use_tracking': False}
)

## Run GUI ##
## ======= ##

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    gui = RealSenseGUI(**recording_params)
    gui.show()
    sys.exit(app.exec())