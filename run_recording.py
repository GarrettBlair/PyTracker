import sys
import logging
from PySide6.QtWidgets import QApplication
from gui import RealSenseGUI
from runtime_camera import select_camera
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
        selected_camera = select_camera(
            preselected_id=recording_params.get('serial_number'),
            preview=True,
            width=recording_params.get('width', 640),
            height=recording_params.get('height', 480),
            fps=recording_params.get('fps', 30),
        )

        if selected_camera['camera_type'] != 'realsense':
            logger.warning(
                "Selected camera is USB. Recording GUI currently supports RealSense only, so only preview was run."
            )
            print(
                "Selected camera is USB. Recording GUI currently supports RealSense only, "
                "so only preview was run."
            )
            sys.exit(0)

        recording_params['serial_number'] = selected_camera['serial_number']
        logger.info("Launching recording GUI with serial=%s", recording_params['serial_number'])

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        gui = RealSenseGUI(**recording_params)
        gui.show()
        exit_code = app.exec()
        logger.info("Recording GUI exited with code=%s", exit_code)
        sys.exit(exit_code)
    except Exception:
        logger.exception("Fatal error while launching recording GUI")
        raise
    finally:
        shutdown_app_logging()