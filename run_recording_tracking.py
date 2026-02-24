import sys
import os
from PySide6.QtWidgets import QApplication
from gui import RealSenseGUI
from experiment_config import load_recording_params

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'recording_config.json')
recording_params = load_recording_params(
    CONFIG_PATH,
    defaults={
        'use_tracking': True,
        'ref_num_frames': 30,
        'tracking_method': 'dark',
        'use_window': True,
        'window_size': 100,
        'window_weight': 0.90,
        'loc_thresh': 99.5,
        'ksize': 5,
    }
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