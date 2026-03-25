import os
import json
import csv
import logging
import cv2
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import ndimage

from PySide6.QtCore import QThread, Signal, Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QHBoxLayout,
    QMessageBox,
)
from PySide6.QtGui import QImage, QPixmap, QFont


class USBCamera(QThread):
    # signal to send the frame and stats back to the GUI for display
    image_data = Signal(np.ndarray)
    stats_data = Signal(int, str)
    elapsed_time = Signal(float)
    reference_ready = Signal(bool)
    tracking_data = Signal(np.ndarray, tuple)

    def __init__(
        self,
        use_tracking=False,
        ref_num_frames=30,
        recording_length=None,
        camera_index=0,
        folder_path=".",
        frames_per_file=1000,
        width=640,
        height=480,
        fps=30,
        codec="XVID",
        exposure=None,
        gain=None,
        tracking_method="dark",
        use_window=True,
        window_size=50,
        window_weight=0.90,
        loc_thresh=99.5,
        ksize=5,
        **_unused_kwargs):
        super().__init__()
        self.recording = False
        self.running = True

        # initialize state variables
        self.writer = None
        self.file_counter = 0
        self.frame_counter = 0
        self.total_frames = 0
        self.recording_start_time = None
        self.capture = None

        # assign provided args as attrs
        self.use_tracking = use_tracking
        self.ref_num_frames = ref_num_frames
        self.recording_length = recording_length
        self.camera_index = int(camera_index)
        self.folder_path = folder_path
        self.recording_path = None
        self.frames_per_file = frames_per_file
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.exposure = exposure
        self.gain = gain

        # assign tracking attributes
        self.tracking_method = tracking_method
        self.use_window = use_window
        self.window_size = window_size
        self.window_weight = window_weight
        self.loc_thresh = loc_thresh
        self.ksize = ksize

        # initialize tracking attributes
        self.computing_reference = False
        self.ref_stack = []
        self.reference = None
        self.prior_position = None
        self.timestamps_csv_path = None
        self.tracking_csv_path = None
        self.session_log_handler = None

        self.logger = logging.getLogger("pytracker.camera")
        self.logger.info(
            "Initialized USBCamera index=%s tracking=%s fps=%s resolution=%sx%s",
            self.camera_index,
            self.use_tracking,
            self.fps,
            self.width,
            self.height,
        )

    def _attach_recording_log_handler(self):
        if self.recording_path is None:
            return

        if self.session_log_handler is not None:
            self.logger.removeHandler(self.session_log_handler)
            self.session_log_handler.close()
            self.session_log_handler = None

        log_path = os.path.join(self.recording_path, "recording.log")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        self.logger.addHandler(handler)
        self.session_log_handler = handler
        self.logger.info("Recording session log initialized at %s", os.path.abspath(log_path))

    def _detach_recording_log_handler(self):
        if self.session_log_handler is not None:
            self.logger.removeHandler(self.session_log_handler)
            self.session_log_handler.close()
            self.session_log_handler = None

    def _open_capture(self):
        # Try DirectShow first on Windows for faster device initialization.
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open USB camera index {self.camera_index}.")

        # Prefer MJPG for USB webcams to reduce USB bandwidth and CPU decode costs.
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(self.codec)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Property support is camera/driver dependent, so set best-effort.
        # Keep USB auto-exposure on by default to avoid accidentally forcing very low FPS.
        if self.exposure is None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))

        if self.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, float(self.gain))

        self.capture = cap
        self.logger.info("USB capture opened successfully")

    def compute_reference(self):
        """
        Generate the reference image for subsequent video tracking.
        """
        self.ref_stack = []
        self.computing_reference = True

    def locate_subject(self, frame):
        """
        Locate the subject in the frame.

        Note: This part of the code is adapted directly from:
        https://github.com/denisecailab/ezTrack/LocationTracking
        """
        frame = frame.astype("int16")
        reference = self.reference.astype("int16")

        # find difference from reference
        if self.tracking_method == "abs":
            diff = np.abs(frame - reference)
        elif self.tracking_method == "light":
            diff = frame - reference
        elif self.tracking_method == "dark":
            diff = reference - frame
        else:
            raise Exception("Invalid 'tracking_method. Must be one of ['abs', 'light', 'dark']")

        # apply window
        if self.prior_position is not None and self.use_window:
            weight = 1 - self.window_weight
            window_size = self.window_size // 2
            ymin, ymax = self.prior_position[0] - window_size, self.prior_position[0] + window_size
            xmin, xmax = self.prior_position[1] - window_size, self.prior_position[1] + window_size

            diff = diff + (diff.min() * -1)  # scale so lowest value is 0
            diff_weights = np.ones(diff.shape) * weight
            diff_weights[slice(int(max(ymin, 0)), int(ymax)), slice(int(max(xmin, 0)), int(xmax))] = 1
            diff = diff * diff_weights

        # threshold differences and find center of mass for remaining values
        diff[diff < np.percentile(diff, self.loc_thresh)] = 0

        # remove influence of wire
        if self.ksize is not None:
            kernel = np.ones((self.ksize, self.ksize), np.uint8)
            diff_morph = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
            krn_violation = diff_morph.sum() == 0
            diff = diff if krn_violation else diff_morph
            if krn_violation:
                self.logger.warning(
                    "ksize too large for morphology at frame=%s; using unfiltered diff",
                    self.total_frames,
                )

        com = ndimage.center_of_mass(diff)  # returns as (y,x)
        return diff, com

    def run(self):
        """
        Start camera and while recording is enabled (triggered externally),
        capture, display, and save frames. Display time elapsed as well.
        """
        try:
            self._open_capture()

            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                # Tracking and writer both expect grayscale data.
                if frame.ndim == 3:
                    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_np = frame

                if self.computing_reference:
                    self.ref_stack.append(frame_np.copy())
                    if len(self.ref_stack) >= self.ref_num_frames:
                        self.reference = np.median(np.array(self.ref_stack), axis=0).astype(np.uint8)
                        self.computing_reference = False
                        self.reference_ready.emit(True)
                        self.logger.info("Reference computed successfully with %s frames", self.ref_num_frames)

                self.image_data.emit(frame_np)  # display frame

                if self.use_tracking and self.reference is not None:
                    diff, com = self.locate_subject(frame_np)
                    self.prior_position = com
                    self.tracking_data.emit(diff, com)
                    if self.recording:
                        with open(self.tracking_csv_path, "a", newline="") as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([self.total_frames, com[1], com[0]])

                if self.recording:
                    self.write_frame(frame_np)

                    # emit time data
                    elapsed = time.time() - self.recording_start_time
                    time_str = str(timedelta(seconds=int(elapsed)))
                    self.stats_data.emit(self.total_frames, time_str)
                    self.elapsed_time.emit(elapsed)

                    # save software timestamp and frame index as the hardware-frame placeholder.
                    timestamp_ms = time.time() * 1000.0
                    with open(self.timestamps_csv_path, "a", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([self.total_frames - 1, timestamp_ms, self.total_frames - 1])
                else:
                    self.stats_data.emit(0, "00:00:00")

        except Exception:
            self.logger.exception("Unhandled exception in camera run loop")
            raise
        finally:
            self.stop_recording()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
                self.logger.info("USB capture released")

    def write_frame(self, frame):
        """
        Write current frame to the currently open AVI file and advance
        frame counter. If the file is full, rotate to next file.
        """
        if self.writer is None or self.frame_counter >= self.frames_per_file:
            self.rotate_file()

        self.writer.write(frame)
        self.frame_counter += 1
        self.total_frames += 1

    def rotate_file(self):
        """
        Release AVI file currently being written and make new file.
        """
        if self.writer:
            self.writer.release()
            self.file_counter += 1
            self.frame_counter = 0

        filename = os.path.join(self.recording_path, f"{self.file_counter}.avi")
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            filename,
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=False,
        )
        self.logger.info("Recording to file: %s", os.path.abspath(filename))

    def start_recording(self):
        """
        Initialize frame and file counters, note current time,
        make file directory if it does not exist, and switch recording flag to True.
        """
        if self.recording:
            return

        self.file_counter = 0
        self.frame_counter = 0
        self.total_frames = 0
        self.recording_start_time = time.time()

        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.recording_path = os.path.join(self.folder_path, current_time)
        if not os.path.exists(self.recording_path):
            os.makedirs(self.recording_path)

        self._attach_recording_log_handler()

        params_keys = [
            "recording_start_time",
            "recording_length",
            "camera_index",
            "folder_path",
            "recording_path",
            "frames_per_file",
            "width",
            "height",
            "fps",
            "codec",
            "exposure",
            "gain",
            "use_tracking",
        ]
        params_info = {key: getattr(self, key) for key in params_keys}
        # params_info["camera_type"] = "usb"
        with open(os.path.join(self.recording_path, "params.json"), "w", encoding="utf-8") as file:
            json.dump(params_info, file, indent=4)
        self.logger.info("Recording started with params: %s", json.dumps(params_info, default=str))

        self.timestamps_csv_path = os.path.join(self.recording_path, "timestamps.csv")
        with open(self.timestamps_csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Frame", "Timestamp", "HW_Frame"])

        if self.use_tracking and self.reference is not None:
            cv2.imwrite(os.path.join(self.recording_path, "reference.png"), self.reference)
            self.tracking_csv_path = os.path.join(self.recording_path, "tracking.csv")
            with open(self.tracking_csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Frame", "X", "Y"])

        self.recording = True

    def stop_recording(self):
        """
        Switch off recording and release AVI file currently being written.
        """
        self.recording = False

        if self.writer:
            self.writer.release()
            self.writer = None
            self.logger.info("Recording stopped.")

        timestamps_df = None
        if self.timestamps_csv_path and os.path.exists(self.timestamps_csv_path):
            try:
                timestamps_df = pd.read_csv(self.timestamps_csv_path)
                if len(timestamps_df) > 0:
                    timestamps_df["TimestampFromStart"] = (
                        timestamps_df["Timestamp"] - timestamps_df["Timestamp"].iloc[0]
                    )
                    timestamps_df.to_csv(self.timestamps_csv_path, index=False)
            except Exception as error:
                self.logger.error("Error processing timestamps: %s", error)

        if self.use_tracking and self.tracking_csv_path and os.path.exists(self.tracking_csv_path):
            try:
                tracking_df = pd.read_csv(self.tracking_csv_path)
                if len(tracking_df) > 0:
                    xy = tracking_df[["X", "Y"]].values
                    xy_diffs = xy[1:] - xy[:-1]
                    distances = np.nan_to_num(np.linalg.norm(xy_diffs, axis=1))
                    tracking_df["Distance_px"] = np.append(0, distances)
                    tracking_df.to_csv(self.tracking_csv_path, index=False)
            except Exception as error:
                self.logger.error("Error processing tracking data: %s", error)

        if self.recording_path is not None:
            elapsed = 0 if self.recording_start_time is None else (time.time() - self.recording_start_time)
            self.logger.info(
                "Recording session ended path=%s total_frames=%s elapsed_s=%.2f",
                os.path.abspath(self.recording_path),
                self.total_frames,
                elapsed,
            )

        self._detach_recording_log_handler()

    def stop_camera(self):
        """
        Stop running camera (used when GUI closes).
        """
        self.running = False
        self.wait()


class USB_GUI(QMainWindow):
    def __init__(
        self,
        use_tracking=False,
        recording_length=None,
        **camera_kwargs,
    ):
        """
        Initialize GUI elements for recording and optional real-time tracking.
        """
        super().__init__()
        self.setWindowTitle("USB Camera Recorder")

        self.use_tracking = use_tracking
        window_width = 1300 if use_tracking else 700
        self.setFixedSize(window_width, 600)

        self.recording_length = np.inf if (recording_length <= 0) or (recording_length is None) else recording_length

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        self.stats_layout = QHBoxLayout()
        self.time_label = QLabel("Duration: 00:00:00")
        self.count_label = QLabel("Frames: 0")
        self.fps_label = QLabel("FPS: --.-   ")
        stat_font = QFont("Arial", 14, QFont.Bold)
        self.time_label.setFont(stat_font)
        self.count_label.setFont(stat_font)
        self.fps_label.setFont(stat_font)
        self.time_label.setStyleSheet("color: #2ecc71;")
        self.stats_layout.addWidget(self.time_label)
        self.stats_layout.addStretch()
        self.stats_layout.addWidget(self.fps_label)
        self.stats_layout.addWidget(self.count_label)
        self.layout.addLayout(self.stats_layout)

        self.fps_measure_start = time.perf_counter()
        self.fps_measure_count = 0
        self.smoothed_fps = None

        self.video_container = QHBoxLayout()

        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.video_container.addWidget(self.video_label)

        if self.use_tracking:
            self.tracking_label = QLabel("Real-time Tracking")
            self.tracking_label.setAlignment(Qt.AlignCenter)
            self.tracking_label.setStyleSheet("background-color: black; border: 2px solid red;")
            self.video_container.addWidget(self.tracking_label)

        self.layout.addLayout(self.video_container)

        self.btn_layout = QHBoxLayout()

        if self.use_tracking:
            self.reference_btn = QPushButton("Compute Reference")
            self.reference_btn.clicked.connect(self.on_reference_clicked)
            self.btn_layout.addWidget(self.reference_btn)

        self.record_btn = QPushButton("Record")
        self.stop_btn = QPushButton("Stop Recording")
        self.record_btn.setEnabled(not self.use_tracking)
        self.stop_btn.setEnabled(False)

        self.btn_layout.addWidget(self.record_btn)
        self.btn_layout.addWidget(self.stop_btn)
        self.layout.addLayout(self.btn_layout)

        self.setCentralWidget(self.central_widget)
        self.record_btn.clicked.connect(self.on_record_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)

        self.blink_timer = QTimer()
        self.blink_timer.setInterval(500)
        self.blink_timer.timeout.connect(self.toggle_light_blink)
        self.blink_state = False
        self.rec_indicator = QLabel()
        self.rec_indicator.setFixedSize(20, 20)
        self.rec_indicator.setStyleSheet("background-color: #444; border-radius: 10px; border: 1px solid #222;")
        self.stats_layout.addWidget(self.rec_indicator)

        self.camera = USBCamera(
            use_tracking=self.use_tracking,
            recording_length=self.recording_length,
            **camera_kwargs,
        )
        self.camera.image_data.connect(self.update_image)
        self.camera.stats_data.connect(self.update_stats)
        self.camera.elapsed_time.connect(self.check_recording_length)
        if self.use_tracking:
            self.camera.tracking_data.connect(self.update_tracking_display)
            self.camera.reference_ready.connect(self.save_reference)
        self.camera.start()

    @Slot(np.ndarray)
    def update_image(self, frame):
        """
        Update displayed image.
        """
        height, width = frame.shape
        bytes_per_line = width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

        self.fps_measure_count += 1
        elapsed = time.perf_counter() - self.fps_measure_start
        if elapsed >= 0.5:
            instant_fps = self.fps_measure_count / elapsed
            if self.smoothed_fps is None:
                self.smoothed_fps = instant_fps
            else:
                self.smoothed_fps = (0.2 * self.smoothed_fps) + (0.8 * instant_fps)

            self.fps_label.setText(f"FPS: {self.smoothed_fps:.1f}")
            self.fps_measure_start = time.perf_counter()
            self.fps_measure_count = 0

    @Slot(int, str)
    def update_stats(self, count, elapsed_time):
        """
        Update elapsed frame count and duration labels.
        """
        self.count_label.setText(f"Frames: {count}")
        self.time_label.setText(f"Duration: {elapsed_time}")

    @Slot(np.ndarray, tuple)
    def update_tracking_display(self, diff, com):
        """
        Given subject tracking data, update its display for the current frame.
        """
        diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
        diff_color = cv2.cvtColor(diff.astype("uint8"), cv2.COLOR_GRAY2BGR)

        if com and not np.isnan(com[0]):
            y, x = int(com[0]), int(com[1])
            cv2.drawMarker(diff_color, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        h, w, ch = diff_color.shape
        q_img = QImage(diff_color.data, w, h, w * ch, QImage.Format_BGR888)
        self.tracking_label.setPixmap(QPixmap.fromImage(q_img))

    def toggle_light_blink(self):
        """
        Turn on blinking red light when recording; turn back to grey when not recording.
        """
        if self.blink_state:
            self.rec_indicator.setStyleSheet("background-color: #444; border-radius: 10px; border: 1px solid #222;")
        else:
            self.rec_indicator.setStyleSheet("background-color: #ff0000; border-radius: 10px; border: 1px solid #a00;")
        self.blink_state = not self.blink_state

    @Slot(bool)
    def save_reference(self, reference_ready):
        """
        When the reference image is ready, save it.
        """
        self.reference_btn.setText("Reference Set")
        self.reference_btn.setEnabled(True)
        self.record_btn.setEnabled(True)

        if reference_ready and self.camera.reference is not None:
            cv2.imwrite(os.path.join(self.camera.folder_path, "reference.png"), self.camera.reference)

    def on_reference_clicked(self):
        """
        When clicked, compute reference and save it as png.
        """
        self.reference_btn.setText("Computing...")
        self.reference_btn.setEnabled(False)
        self.camera.compute_reference()

    def on_record_clicked(self):
        """
        Disable record button, start blink indicator, and start recording.
        """
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.blink_timer.start()
        self.rec_indicator.setStyleSheet("background-color: #ff0000; border-radius: 10px; border: 1px solid #a00;")
        self.blink_state = True

        self.camera.start_recording()

    def on_stop_clicked(self):
        """
        Disable stop button, stop blink indicator, and stop recording.
        """
        self.camera.stop_recording()

        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.blink_timer.stop()
        self.rec_indicator.setStyleSheet("background-color: #444; border-radius: 10px; border: 1px solid #222;")
        self.blink_state = False

        msg = QMessageBox(self)
        msg.setText(f"Recording Ended. Data recorded at<br>{os.path.abspath(self.camera.recording_path)}.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    @Slot(float)
    def check_recording_length(self, elapsed):
        """
        Stop recording when requested recording length has elapsed.
        """
        if elapsed >= self.recording_length:
            self.on_stop_clicked()

    def closeEvent(self, event):
        """
        When GUI is closed, stop camera thread.
        """
        self.camera.stop_camera()
        super().closeEvent(event)
