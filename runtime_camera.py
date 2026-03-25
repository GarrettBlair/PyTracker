import cv2
import numpy as np
import logging

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


logger = logging.getLogger("pytracker.camera_select")


def _is_highgui_not_available_error(error):
    message = str(error).lower()
    return (
        "the function is not implemented" in message
        and ("cvnamedwindow" in message or "highgui" in message or "imshow" in message)
    )


def _warn_preview_unavailable(error):
    message = (
        "OpenCV preview is unavailable in this environment (HighGUI is not built/enabled). "
        "Skipping preview and continuing."
    )
    logger.warning("%s Details: %s", message, error)
    print(f"WARNING: {message}")


def list_realsense_cameras():
    """
    Return metadata for all connected RealSense cameras.
    """
    if rs is None:
        return []

    context = rs.context()
    cameras = []

    for device in context.query_devices():
        serial_number = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        product_line = ""
        try:
            product_line = device.get_info(rs.camera_info.product_line)
        except Exception:
            product_line = ""

        cameras.append(
            {
                "serial_number": serial_number,
                "name": name,
                "product_line": product_line,
            }
        )

    return cameras


def list_usb_cameras(preselected_index=None, max_index=10):
    """
    Return metadata for USB cameras discoverable by OpenCV.
    """
    cameras = []
    
    if preselected_index is not None:
        start_idx = preselected_index
        max_index = preselected_index + 1
    else:
        start_idx = 0
    
    for index in range(start_idx, max_index):
        print(f"Checking for USB camera at index {index}...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        if ok:
            cameras.append(
                {
                    "camera_type": "usb",
                    "name": f"USB Camera {index}",
                    "id": str(index),
                    "index": index,
                }
            )
        cap.release()

    return cameras


def list_connected_cameras(max_usb_index=10):
    """
    Return metadata for all detected cameras (RealSense + USB).
    """
    cameras = []
    for camera in list_realsense_cameras():
        cameras.append(
            {
                "camera_type": "realsense",
                "name": camera["name"],
                "id": camera["serial_number"],
                "serial_number": camera["serial_number"],
                "product_line": camera.get("product_line", ""),
            }
        )

    cameras.extend(list_usb_cameras(max_index=max_usb_index))
    return cameras

def select_camera(device_id=None, max_usb_index=3):
    """
    Select a connected camera at runtime and optionally preview it.

    Returns
    =======
    dict
        Selected camera metadata.
    """
    if device_id is not None:
        cameras = list_usb_cameras(preselected_index=device_id, max_index=device_id+3)
    else:
        cameras = list_connected_cameras(max_usb_index=max_usb_index)

    if len(cameras) == 0:
        raise RuntimeError("No cameras were detected (RealSense or USB).")

    selected_choice_idx = None

    if device_id is not None:
        selected = None
        device_id_str = str(device_id)
        for camera in cameras:
            if camera["id"] == device_id_str:
                selected = camera
                break
        if selected is None:
            raise RuntimeError(
                f"Requested camera id {device_id} was not found among connected cameras."
            )
    elif len(cameras) == 1:
        selected = cameras[0]
        logger.info(
            "Detected one camera: [%s] %s (%s).",
            selected['camera_type'],
            selected['name'],
            selected['id'],
        )
        print(f"Detected one camera: [{selected['camera_type']}] {selected['name']} ({selected['id']}).")
    else:
        logger.info("Connected cameras:")
        print("Connected cameras:")
        for idx, camera in enumerate(cameras):
            extra = ""
            if camera["camera_type"] == "realsense" and camera.get("product_line"):
                extra = f" - {camera['product_line']}"
            logger.info(
                "  [%s] [%s] %s (%s)%s",
                idx,
                camera['camera_type'],
                camera['name'],
                camera['id'],
                extra,
            )
            print(f"  [{idx}] [{camera['camera_type']}] {camera['name']} ({camera['id']}){extra}")

        while True:
            choice = input(f"Select camera [0-{len(cameras)-1}]: ").strip()
            if choice.isdigit() and 0 <= int(choice) < len(cameras):
                selected_choice_idx = int(choice)
                selected = cameras[selected_choice_idx]
                break
            logger.warning("Invalid selection. Please enter a valid number.")
            print("Invalid selection. Please enter a valid number.")

    if selected.get("camera_type") == "usb" and "index" not in selected:
        if selected_choice_idx is not None:
            selected["index"] = selected_choice_idx
        else:
            selected["index"] = int(selected["id"])

    logger.info(
        "Selected camera: [%s] %s (%s)",
        selected['camera_type'],
        selected['name'],
        selected['id'])
    print(f"Selected camera: [{selected['camera_type']}] {selected['name']} ({selected['id']})")

    return selected


def select_realsense_camera(preselected_serial=None):
    """
    Select a connected RealSense camera at runtime and optionally preview it.

    Returns
    =======
    dict
        Selected RealSense camera metadata.
    """
    cameras = [cam for cam in list_connected_cameras() if cam["camera_type"] == "realsense"]
    if len(cameras) == 0:
        raise RuntimeError("No RealSense cameras were detected.")

    if preselected_serial:
        matches = [cam for cam in cameras if cam["serial_number"] == preselected_serial]
        if not matches:
            raise RuntimeError(
                f"Requested serial number {preselected_serial} was not found among connected cameras."
            )
        selected = matches[0]
    elif len(cameras) == 1:
        selected = cameras[0]
        logger.info("Detected one camera: %s (%s).", selected['name'], selected['serial_number'])
        print(f"Detected one camera: {selected['name']} ({selected['serial_number']}).")
    else:
        logger.info("Connected RealSense cameras:")
        print("Connected RealSense cameras:")
        for idx, camera in enumerate(cameras):
            product_line = f" - {camera['product_line']}" if camera["product_line"] else ""
            logger.info("  [%s] %s (%s)%s", idx, camera['name'], camera['serial_number'], product_line)
            print(f"  [{idx}] {camera['name']} ({camera['serial_number']}){product_line}")

        while True:
            choice = input(f"Select camera [0-{len(cameras)-1}]: ").strip()
            if choice.isdigit() and 0 <= int(choice) < len(cameras):
                selected = cameras[int(choice)]
                break
            logger.warning("Invalid selection. Please enter a valid number.")
            print("Invalid selection. Please enter a valid number.")

    selected_serial = selected["serial_number"]
    logger.info("Selected camera: %s (%s)", selected['name'], selected_serial)
    print(f"Selected camera: {selected['name']} ({selected_serial})")

    return {
        "camera_type": "realsense",
        "name": selected["name"],
        "id": selected_serial,
        "serial_number": selected_serial,
        "product_line": selected.get("product_line", ""),
    }
