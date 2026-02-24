import logging
from runtime_camera import select_camera
from logging_config import setup_app_logging, shutdown_app_logging

preview_params = {
    'camera_id': None,      # set to serial (RealSense) or index string (USB), e.g. '213622074070' or '0'
    'width': 640,
    'height': 480,
    'fps': 30,
    'max_usb_index': 10,
}


if __name__ == "__main__":
    setup_app_logging()
    logger = logging.getLogger("pytracker.launch")

    try:
        select_camera(
            preselected_id=preview_params.get('camera_id'),
            preview=True,
            width=preview_params.get('width', 640),
            height=preview_params.get('height', 480),
            fps=preview_params.get('fps', 30),
            max_usb_index=preview_params.get('max_usb_index', 10),
        )
        logger.info("Camera preview ended")
    except Exception:
        logger.exception("Fatal error while running camera preview")
        raise
    finally:
        shutdown_app_logging()
