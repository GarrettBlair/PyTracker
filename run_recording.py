import sys
import logging
from PySide6.QtWidgets import QApplication
from gui import RealSenseGUI
from gui_usb import USB_GUI
from runtime_camera import select_camera, select_realsense_camera
from logging_config import setup_app_logging, shutdown_app_logging

recording_params = {
    'serial_number'   : None, # leave as None to load the first available one
    'folder_path'     : './data/',
    'recording_length': 10, # in seconds, leave 0 if you want recording indefinitely
    'frames_per_file' : 1000,
    'width'       : 640,
    'height'      : 480,
    'fps'         : 30,
    'codec'       : 'XVID',
    'exposure'    : 2500,
    'gain'        : 50,
    'laser_power' : 200,
    'enable_ttl'  : True,
    'use_tracking': False
}

## Run GUI ##
## ======= ##

if __name__ == "__main__":
    setup_app_logging(console_output=False)
    logger = logging.getLogger("pytracker.launch")

    try:
        if recording_params.get('serial_number') is not None:
            logger.info("Preselected camera serial number: %s", recording_params['serial_number'])
            selected_camera = select_realsense_camera(preselected_serial=recording_params['serial_number'])
        else:
            logger.info("No preselected camera. Will prompt user to select from detected cameras.")
            selected_camera = select_camera(max_usb_index=recording_params.get('max_devices', 5))
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        if selected_camera['camera_type'] == 'realsense':
            recording_params['serial_number'] = selected_camera['serial_number']
            logger.info("Launching recording GUI with RealSense serial=%s", recording_params['serial_number'])
            gui = RealSenseGUI(**recording_params)
        else:
            recording_params['camera_index'] = selected_camera['index']
            recording_params['exposure'] = None
            recording_params['laser_power'] = None
            recording_params['enable_ttl'] = False
            logger.info("Launching recording GUI with USB index=%s", recording_params['camera_index'])
            gui = USB_GUI(**recording_params)
        gui.show()
        exit_code = app.exec()
        logger.info("Recording GUI exited with code=%s", exit_code)
        sys.exit(exit_code)
    except Exception:
        logger.exception("Fatal error while launching recording GUI")
        raise
    finally:
        shutdown_app_logging()