import sys
import logging
from PySide6.QtWidgets import QApplication
from gui import RealSenseGUI
from gui_usb import USB_GUI
from runtime_camera import select_camera, select_realsense_camera
from logging_config import setup_app_logging, shutdown_app_logging

#### USB CAMERA PARAMS ####
# recording_params = {
#     'serial_number'   : None, # leave as None to load the first available one
#     'folder_path'     : './data/',
#     'recording_length': 10, # in seconds, leave 0 if you want recording indefinitely
#     'frames_per_file' : 1000,
#     'width'       : 640,
#     'height'      : 480,
#     'fps'         : 30,
#     'codec'       : 'MJPG', # MJPG
#     'exposure'    : None, # setting exposure currently not working for USB cameras, so set to None to use camera default
#     'gain'        : 50, # has no change currently
#     'laser_power' : None,
#     'enable_ttl'  : False,
#     'device_index': None, # for USB cameras, this is the camera index (0, 1, etc.)
#     'max_devices' : 4, # for camera selection, set to None to show all detected cameras
    
#     # tracking parameters
#     'use_tracking'    : True,
#     'ref_num_frames'  : 30,
#     'tracking_method' : 'dark',
#     'use_window'      : True,
#     'window_size'     : 100,
#     'window_weight'   : 0.90,
#     'loc_thresh'      : 99.5,
#     'ksize'           : 5,
# }
#### REALSENSE PARAMS ####
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
    # tracking parameters
    'use_tracking'    : True,
    'ref_num_frames'  : 30,
    'tracking_method' : 'dark',
    'use_window'      : True,
    'window_size'     : 100,
    'window_weight'   : 0.90,
    'loc_thresh'      : 99.5,
    'ksize'           : 5,
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

        elif recording_params.get('device_index') is not None:
            # from runtime_camera import select_camera
            selected_camera = select_camera(device_id=recording_params.get('device_index'))
            logger.info("Preselected camera device index: %s", recording_params['device_index'])
        else:
            logger.info("No preselected camera. Will prompt user to select from detected cameras.")
            selected_camera = select_camera( max_usb_index=recording_params.get('max_devices', 5) )

        # recording_params['camera_type'] = selected_camera['camera_type']
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        if selected_camera['camera_type'] == 'realsense':
            recording_params['serial_number'] = selected_camera['serial_number']
            logger.info("Launching tracking GUI with RealSense serial=%s", recording_params['serial_number'])
            gui = RealSenseGUI(**recording_params)
        else:
            recording_params['camera_index'] = selected_camera['index']
            # recording_params['exposure'] = None
            recording_params['laser_power'] = None
            recording_params['enable_ttl'] = False
            logger.info("Launching tracking GUI with USB index=%s", recording_params['camera_index'])
            gui = USB_GUI(**recording_params)
        gui.show()
        exit_code = app.exec()
        logger.info("Tracking GUI exited with code=%s", exit_code)
        sys.exit(exit_code)
    except Exception:
        logger.exception("Fatal error while launching tracking GUI")
        raise
    finally:
        shutdown_app_logging()