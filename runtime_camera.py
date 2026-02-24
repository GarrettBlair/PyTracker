import cv2
import numpy as np
import logging

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


logger = logging.getLogger("pytracker.camera_select")


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


def list_usb_cameras(max_index=10):
    """
    Return metadata for USB cameras discoverable by OpenCV.
    """
    cameras = []
    for index in range(max_index):
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


def preview_realsense_camera(serial_number, width=640, height=480, fps=30, window_name=None):
    """
    Open a live preview for the selected RealSense camera until q or ESC is pressed.
    """
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed, so RealSense preview is unavailable.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)

    if window_name is None:
        window_name = f"RealSense Preview - {serial_number}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        pipeline.start(config)
        while True:
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame()
            if not ir_frame:
                continue

            frame_np = np.asanyarray(ir_frame.get_data())
            cv2.imshow(window_name, frame_np)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyWindow(window_name)


def preview_usb_camera(camera_index, width=640, height=480, fps=30, window_name=None):
    """
    Open a live preview for a USB camera until q or ESC is pressed.
    """
    if window_name is None:
        window_name = f"USB Camera Preview - index {camera_index}"

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open USB camera index {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if frame.ndim == 3:
                frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_show = frame

            cv2.imshow(window_name, frame_show)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def preview_camera(camera_info, width=640, height=480, fps=30):
    """
    Preview any camera returned by list_connected_cameras.
    """
    if camera_info["camera_type"] == "realsense":
        preview_realsense_camera(
            serial_number=camera_info["serial_number"],
            width=width,
            height=height,
            fps=fps,
        )
    elif camera_info["camera_type"] == "usb":
        preview_usb_camera(
            camera_index=camera_info["index"],
            width=width,
            height=height,
            fps=fps,
        )
    else:
        raise RuntimeError(f"Unsupported camera type: {camera_info['camera_type']}")


def select_camera(preselected_id=None, preview=True, width=640, height=480, fps=30, max_usb_index=10):
    """
    Select a connected camera at runtime and optionally preview it.

    Returns
    =======
    dict
        Selected camera metadata.
    """
    cameras = list_connected_cameras(max_usb_index=max_usb_index)
    if len(cameras) == 0:
        raise RuntimeError("No cameras were detected (RealSense or USB).")

    if preselected_id is not None:
        selected = None
        preselected_id_str = str(preselected_id)
        for camera in cameras:
            if camera["id"] == preselected_id_str:
                selected = camera
                break
        if selected is None:
            raise RuntimeError(
                f"Requested camera id {preselected_id} was not found among connected cameras."
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
        for idx, camera in enumerate(cameras, start=1):
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
            choice = input(f"Select camera [1-{len(cameras)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(cameras):
                selected = cameras[int(choice) - 1]
                break
            logger.warning("Invalid selection. Please enter a valid number.")
            print("Invalid selection. Please enter a valid number.")

    logger.info(
        "Selected camera: [%s] %s (%s)",
        selected['camera_type'],
        selected['name'],
        selected['id'],
    )
    print(f"Selected camera: [{selected['camera_type']}] {selected['name']} ({selected['id']})")

    if preview:
        logger.info("Starting preview. Press 'q' or ESC to close preview and continue.")
        print("Starting preview. Press 'q' or ESC to close preview and continue.")
        preview_camera(selected, width=width, height=height, fps=fps)

    return selected


def select_realsense_camera(
    preselected_serial=None,
    preview=True,
    width=640,
    height=480,
    fps=30,
):
    """
    Select a connected RealSense camera at runtime and optionally preview it.

    Returns
    =======
    str
        The selected serial number.
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
        for idx, camera in enumerate(cameras, start=1):
            product_line = f" - {camera['product_line']}" if camera["product_line"] else ""
            logger.info("  [%s] %s (%s)%s", idx, camera['name'], camera['serial_number'], product_line)
            print(f"  [{idx}] {camera['name']} ({camera['serial_number']}){product_line}")

        while True:
            choice = input(f"Select camera [1-{len(cameras)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(cameras):
                selected = cameras[int(choice) - 1]
                break
            logger.warning("Invalid selection. Please enter a valid number.")
            print("Invalid selection. Please enter a valid number.")

    selected_serial = selected["serial_number"]
    logger.info("Selected camera: %s (%s)", selected['name'], selected_serial)
    print(f"Selected camera: {selected['name']} ({selected_serial})")

    if preview:
        logger.info("Starting preview. Press 'q' or ESC to close preview and continue.")
        print("Starting preview. Press 'q' or ESC to close preview and continue.")
        preview_realsense_camera(
            serial_number=selected_serial,
            width=width,
            height=height,
            fps=fps,
        )

    return selected_serial
