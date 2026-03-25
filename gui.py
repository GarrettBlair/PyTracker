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
import pyrealsense2 as rs

from PySide6.QtCore import QThread, Signal, Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QHBoxLayout,
    QMessageBox
    )
from PySide6.QtGui import QImage, QPixmap, QFont


## Class to engage RealSense camera ##
## ================================ ##

class RealSenseCamera(QThread):
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
            serial_number=None,
            folder_path='.',
            frames_per_file=1000,
            width=640,
            height=480,
            fps=30,
            codec='XVID',
            exposure=5000,
            gain=50,
            laser_power=150,
            enable_ttl=True,
            tracking_method='dark',
            use_window=True,
            window_size=50,
            window_weight=0.90,
            loc_thresh=99.5,
            ksize=5,
            **_unused_kwargs,
            ):
        '''
        Initialize RealSense camera, define recording
        parameters, and enable infrared stream.

        Parameters
        ==========
        use_tracking : bool
            Whether or not to use real-time tracking. Default is False.
        recording_length : int
            Time in seconds to make the recording. If 0 or negative or None, will be converted to np.inf and
            the recording will run until the recording is manually stopped. Default is None.
        serial_number : str
            Serial number of the desired camera to enable. If None,
            default camera will be enabled. Default is None.
        folder_path : str
            Directory where the files will be stored. A folder with
            the date and time will be created there. Default is '.'.
        frames_per_file : int
            Number of frames that each avi file will store. Default is 1000.
        width, height : int
            Width and height of the FOV of the camera. Defaults are 640 and
            480, respectively.
        fps : int
            Desired frame rate of the camera. Default is 30.
        codec : str
            Which codec to use to save avi files. Default is 'XVID'.
        exposure : int
            Exposure of the camera stream. Default is 5000.
        gain : int
            Digital gain of the camera stream. Default is 50.
        laser_power : int
            Power of the infrared laser in the camera. Default is 150.
        enable_ttl : bool
            Whether or not to use output trigger on every frame. Default is True.
        '''
        super().__init__()
        self.recording = False
        self.running = True
        
        # initialize state variables
        self.writer = None
        self.file_counter = 0
        self.frame_counter = 0
        self.total_frames = 0
        self.recording_start_time = None

        # assign provided args as attrs
        self.use_tracking = use_tracking
        self.ref_num_frames = ref_num_frames
        self.recording_length = recording_length
        self.serial_number = serial_number
        self.folder_path = folder_path
        self.recording_path = None
        self.frames_per_file = frames_per_file
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.exposure = exposure
        self.gain = gain
        self.laser_power = laser_power
        self.enable_ttl = enable_ttl

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
            "Initialized RealSenseCamera serial=%s tracking=%s fps=%s resolution=%sx%s",
            self.serial_number,
            self.use_tracking,
            self.fps,
            self.width,
            self.height,
        )
        
        # set up realsense stream
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.serial_number is not None:
            self.config.enable_device(serial_number)
        self.config.enable_stream(
            rs.stream.infrared,
            1,
            width,
            height,
            rs.format.y8,
            fps
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

    def compute_reference(self):
        '''
        Generate the reference image for subsequent video tracking
        '''
        self.ref_stack = []
        self.computing_reference = True
    
    def locate_subject(self, frame):
        '''
        Locate the subject in the frame.
        
        Note: This part of the code is adapted directly from:
        https://github.com/denisecailab/ezTrack/LocationTracking
        '''
        frame = frame.astype('int16')
        reference = self.reference.astype('int16')
        
        # find difference from reference
        if self.tracking_method == 'abs':
            diff = np.abs(frame - reference)
        elif self.tracking_method == 'light':
            diff = frame - reference
        elif self.tracking_method == 'dark':
            diff = reference - frame
        else:
            raise Exception("Invalid 'tracking_method. Must be one of ['abs', 'light', 'dark']")
    
        # apply window
        if self.prior_position is not None and self.use_window:
            weight = 1 - self.window_weight
            window_size = self.window_size//2
            ymin,ymax = self.prior_position[0]-window_size, self.prior_position[0]+window_size
            xmin,xmax = self.prior_position[1]-window_size, self.prior_position[1]+window_size

            diff = diff + (diff.min() * -1) #scale so lowest value is 0
            diff_weights = np.ones(diff.shape)*weight
            diff_weights[slice(int(max(ymin, 0)), int(ymax)),
                         slice(int(max(xmin, 0)), int(xmax))]=1
            diff = diff*diff_weights
        
        # threshold differences and find center of mass for remaining values
        diff[diff<np.percentile(diff,self.loc_thresh)]=0
        
        # remove influence of wire
        if self.ksize is not None:
            kernel = np.ones((self.ksize,self.ksize),np.uint8)
            diff_morph = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
            krn_violation = diff_morph.sum()==0
            diff = diff if krn_violation else diff_morph
            if krn_violation:
                self.logger.warning(
                    "ksize too large for morphology at frame=%s; using unfiltered diff",
                    self.total_frames,
                )
                print(f"WARNING: ksize too large. Not applying for frame {self.total_frames}")
        
        com = ndimage.center_of_mass(diff) # returns as (y,x)

        return diff, com

    def run(self):
        '''
        Start camera and while recording is enabled (triggered
        externally), capture, display, and save frames. Display
        time elapsed as well.
        '''
        pipeline_started = False
        try:
            profile = self.pipeline.start(self.config)
            pipeline_started = True
            device = profile.get_device()
            # note: depth and infrared are on the same sensor
            self.depth_sensor = device.first_depth_sensor()

            # manual exposure
            self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.depth_sensor.set_option(rs.option.exposure, self.exposure)
            self.depth_sensor.set_option(rs.option.gain, self.gain)
            self.depth_sensor.set_option(rs.option.laser_power, self.laser_power)
            self.logger.info("RealSense pipeline started successfully")

            while self.running:
                try:
                    frames = self.pipeline.wait_for_frames()
                except RuntimeError:
                    self.logger.warning("Frame timeout while waiting for frames; retrying")
                    print("Frame timeout. Retrying...")
                    continue
                
                ir_frame = frames.get_infrared_frame()
                if not ir_frame:
                    continue
                frame_np = np.asanyarray(ir_frame.get_data())

                if self.computing_reference:
                    self.ref_stack.append(frame_np.copy())
                    if len(self.ref_stack) >= self.ref_num_frames:
                        self.reference = np.median(np.array(self.ref_stack), axis=0).astype(np.uint8)
                        self.computing_reference = False
                        self.reference_ready.emit(True)
                        self.logger.info("Reference computed successfully with %s frames", self.ref_num_frames)
                        print("Reference computed successfully.")

                self.image_data.emit(frame_np) # display frame

                if self.use_tracking and self.reference is not None:
                    diff, com = self.locate_subject(frame_np)
                    self.prior_position = com
                    self.tracking_data.emit(diff, com)
                    if self.recording:
                        with open(self.tracking_csv_path, 'a', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([self.total_frames, com[1], com[0]])

                if self.recording:
                    self.write_frame(frame_np)

                    # emit time data
                    elapsed = time.time() - self.recording_start_time
                    time_str = str(timedelta(seconds=int(elapsed)))
                    self.stats_data.emit(self.total_frames, time_str)
                    self.elapsed_time.emit(elapsed)

                    # save timestamp
                    timestamp = ir_frame.get_timestamp()
                    hw_counter = -1
                    if ir_frame.supports_frame_metadata(rs.frame_metadata_value.frame_counter): # hardware frame counter
                        hw_counter = ir_frame.get_frame_metadata(rs.frame_metadata_value.frame_counter)
                    with open(self.timestamps_csv_path, 'a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([self.total_frames-1, timestamp, hw_counter]) # frame was already bumped in write_frame()

                else:
                    self.stats_data.emit(0, "00:00:00")

        except Exception:
            self.logger.exception("Unhandled exception in camera run loop")
            raise
        finally:
            self.stop_recording()
            if pipeline_started:
                self.pipeline.stop()
                self.logger.info("RealSense pipeline stopped")

    def write_frame(self, frame):
        '''
        Write current frame to the currently open AVI file and advance
        frame counter. If the file is full, rotate to next file.
        '''
        if self.writer is None or self.frame_counter >= self.frames_per_file:
            self.rotate_file()
            
        self.writer.write(frame)
        self.frame_counter += 1
        self.total_frames += 1

    def rotate_file(self):
        '''
        Release AVI file currently being written and make new file.
        '''
        if self.writer:
            self.writer.release()
            self.file_counter += 1
            self.frame_counter = 0
        
        # create new compressed AVI file
        filename = os.path.join(self.recording_path, f"{self.file_counter}.avi")
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            filename,
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=False
            )
        self.logger.info("Recording to file: %s", os.path.abspath(filename))
        print(f"Recording to: {filename}")

    def start_recording(self):
        '''
        Initialize frame and file counters, note current time,
        make file directory if it doesn't exist, and switch recording
        flag to True. Enable TTL.
        '''
        if not self.recording:
            self.file_counter = 0
            self.frame_counter = 0
            self.total_frames = 0
            self.recording_start_time = time.time()

            # define folder and create if doesn't exist
            current_time = str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
            self.recording_path = os.path.join(self.folder_path, current_time)
            if not os.path.exists(self.recording_path):
                os.makedirs(self.recording_path)

            self._attach_recording_log_handler()
            
            # save params file of the recording's attributes
            params_keys = ['recording_start_time', 'recording_length', 'serial_number', 'folder_path', 'recording_path',
                        'frames_per_file', 'width', 'height', 'fps', 'codec', 'exposure', 'gain', 'laser_power', 'enable_ttl']
            params_info = {key: getattr(self, key) for key in params_keys}
            with open(os.path.join(self.recording_path, 'params.json'), 'w') as f:
                json.dump(params_info, f, indent=4)
            self.logger.info("Recording started with params: %s", json.dumps(params_info, default=str))
            del params_info

            # create timestamps csv file
            self.timestamps_csv_path = os.path.join(self.recording_path, 'timestamps.csv')
            with open(self.timestamps_csv_path, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Frame', 'Timestamp', 'HW_Frame'])
            
            if self.use_tracking:
                # save reference into recording folder
                cv2.imwrite(
                    os.path.join(self.recording_path, 'reference.png'),
                    self.reference
                    )

                # create tracking csv file
                self.tracking_csv_path = os.path.join(self.recording_path, 'tracking.csv')
                with open(self.tracking_csv_path, 'a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['Frame', 'X', 'Y'])

            # turn on ttl
            if self.enable_ttl:
                if self.depth_sensor.supports(rs.option.output_trigger_enabled):
                    self.depth_sensor.set_option(rs.option.output_trigger_enabled, 1)

            self.recording = True

    def stop_recording(self):
        '''
        Switch off recording and release AVI file currently being
        written. Disable TTL.
        '''
        self.recording = False

        # turn off TTL
        if self.enable_ttl:
            if hasattr(self, 'depth_sensor') and self.depth_sensor.supports(rs.option.output_trigger_enabled):
                self.depth_sensor.set_option(rs.option.output_trigger_enabled, 0)

        if self.writer:
            self.writer.release()
            self.writer = None
            self.logger.info('Recording stopped.')
            print('Recording stopped.')
        
        # re-scale timestamps
        if self.timestamps_csv_path:
            if os.path.exists(self.timestamps_csv_path):
                try:
                    timestamps_df = pd.read_csv(self.timestamps_csv_path)
                    if len(timestamps_df) > 0:
                        timestamps_df['TimestampFromStart'] = timestamps_df['Timestamp'] - timestamps_df['Timestamp'][0]
                        timestamps_df.to_csv(self.timestamps_csv_path, index=False)
                except Exception as e:
                    self.logger.error("Error processing timestamps: %s", e)
                    print(f"Error processing timestamps: {e}")

        # compute distance traveled
        if self.use_tracking and self.tracking_csv_path:
            if os.path.exists(self.tracking_csv_path):
                try:
                    tracking_df = pd.read_csv(self.tracking_csv_path)
                    if len(timestamps_df) > 0:
                        xy = tracking_df[['X','Y']].values
                        xy_diffs = xy[1:] - xy[:-1]
                        distances = np.nan_to_num(np.linalg.norm(xy_diffs, axis=1))
                        tracking_df['Distance_px'] = np.append(0, distances)
                        tracking_df.to_csv(self.tracking_csv_path, index=False)
                except Exception as e:
                    self.logger.error("Error processing tracking data: %s", e)
                    print(f"Error processing tracking data: {e}")

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
        '''
        Stop running camera (used when GUI closes).
        '''
        self.running = False
        self.wait()


## Class to display and operate recording GUI ##
## ========================================== ##

class RealSenseGUI(QMainWindow):
    def __init__(
            self,
            use_tracking=False,
            recording_length=None,
            **camera_kwargs
            ):
        '''
        Initialize GUI elements, with spaces for the video
        feed, time elapsed, a recording indicator light, and
        buttons to begin and end recording. Also initialize
        camera object.

        Parameters
        ==========
        use_tracking : bool
            Whether or not to use real-time tracking. Default is False.
        recording_length : int
            Time in seconds to make the recording. If 0 or negative or None, will be converted to np.inf and
            the recording will run until the recording is manually stopped. Default is None.
        ### See the RealSenseCamera class for the camera_kwargs parameters ###
        '''
        super().__init__()
        self.setWindowTitle("RealSense IR Recorder")

        self.use_tracking = use_tracking # whether or not to use real-time tracking
        window_width = 1300 if use_tracking else 700
        self.setFixedSize(window_width, 600)

        # set variables
        self.recording_length = np.inf if (recording_length<=0)|(recording_length is None) else recording_length

        # ---------------------------------------------
        # UI layout initialization
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        # display stats
        self.stats_layout = QHBoxLayout()
        self.time_label = QLabel("Duration: 00:00:00")
        self.count_label = QLabel("Frames: 0")
        self.fps_label = QLabel("FPS: --.-")
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

        # display video label
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: black; border: 2px solid gray;"
            )
        self.video_container.addWidget(self.video_label)

        # display tracking feed if using
        if self.use_tracking:
            self.tracking_label = QLabel("Real-time Tracking")
            self.tracking_label.setAlignment(Qt.AlignCenter)
            self.tracking_label.setStyleSheet(
                "background-color: black; border: 2px solid red;"
            )
            self.video_container.addWidget(self.tracking_label)
        
        self.layout.addLayout(self.video_container)

        # create and connect buttons
        self.btn_layout = QHBoxLayout()

        if self.use_tracking:
            self.reference_btn = QPushButton("Compute Reference")
            self.reference_btn.clicked.connect(self.on_reference_clicked)
            self.btn_layout.addWidget(self.reference_btn)
        
        self.record_btn = QPushButton("Record")
        self.stop_btn = QPushButton("Stop Recording")
        self.record_btn.setEnabled(not self.use_tracking) # initially disable Record if using tracking
        self.stop_btn.setEnabled(False)
        
        self.btn_layout.addWidget(self.record_btn)
        self.btn_layout.addWidget(self.stop_btn)
        self.layout.addLayout(self.btn_layout)
        
        self.setCentralWidget(self.central_widget)
        self.record_btn.clicked.connect(self.on_record_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)

        # add blinking light when recording
        self.blink_timer = QTimer()
        self.blink_timer.setInterval(500) # every 500ms
        self.blink_timer.timeout.connect(self.toggle_light_blink)
        self.blink_state = False
        self.rec_indicator = QLabel()
        self.rec_indicator.setFixedSize(20, 20)
        self.rec_indicator.setStyleSheet(
            "background-color: #444; border-radius: 10px; border: 1px solid #222;"
            )
        self.stats_layout.addWidget(self.rec_indicator)

        # ---------------------------------------------

        # start camera thread
        self.camera = RealSenseCamera(
            use_tracking=self.use_tracking,
            recording_length=self.recording_length,
            **camera_kwargs
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
        '''
        Update displayed image.
        '''
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
        '''
        Update time elapsed in frames and time.
        '''
        self.count_label.setText(f"Frames: {count}")
        self.time_label.setText(f"Duration: {elapsed_time}")
    
    @Slot(np.ndarray, tuple)
    def update_tracking_display(self, diff, com):
        '''
        Given subject tracking data, update its display for the current frame.
        '''
        
        diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255 # bring to positive values [0,255]
        diff_color = cv2.cvtColor(diff.astype('uint8'), cv2.COLOR_GRAY2BGR)

        # draw COM
        if com and not np.isnan(com[0]):
            y, x = int(com[0]), int(com[1])
            cv2.drawMarker(diff_color, (x,y), (0,0,255), cv2.MARKER_CROSS, 20, 2)
        
        # draw diff
        h, w, ch = diff_color.shape
        q_img = QImage(diff_color.data, w, h, w*ch, QImage.Format_BGR888)
        self.tracking_label.setPixmap(QPixmap.fromImage(q_img))


    def toggle_light_blink(self):
        '''
        Turn on blinking red light when recording; turn back to
        grey when not recording.
        '''
        if self.blink_state:
            self.rec_indicator.setStyleSheet(
                "background-color: #444; border-radius: 10px; border: 1px solid #222;"
                )
        else:
            self.rec_indicator.setStyleSheet(
                "background-color: #ff0000; border-radius: 10px; border: 1px solid #a00;"
                )
        self.blink_state = not self.blink_state
    
    @Slot(bool)
    def save_reference(self, reference_ready):
        '''
        When the reference image is ready, save it.
        '''
        self.reference_btn.setText("Reference Set")
        self.reference_btn.setEnabled(True)
        self.record_btn.setEnabled(True)

        if reference_ready and self.camera.reference is not None:
            cv2.imwrite(
                os.path.join(self.camera.folder_path, 'reference.png'),
                self.camera.reference
                )

    def on_reference_clicked(self):
        '''
        When clicked, compute reference and save it
        as a png to recording folder.
        '''
        self.reference_btn.setText("Computing...")
        self.reference_btn.setEnabled(False)
        self.camera.compute_reference()

    def on_record_clicked(self):
        '''
        Disable record button when clicked, begin
        timer, start blinking light, and start recording.
        '''
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.blink_timer.start()
        self.rec_indicator.setStyleSheet(
            "background-color: #ff0000; border-radius: 10px; border: 1px solid #a00;"
            )
        self.blink_state = True

        self.camera.start_recording()
    
    def on_stop_clicked(self):
        '''
        Disable stop recording button, stop timer,
        stop blinking light, and stop recording.
        '''
        self.camera.stop_recording()

        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.blink_timer.stop()
        self.rec_indicator.setStyleSheet(
            "background-color: #444; border-radius: 10px; border: 1px solid #222;"
            )
        self.blink_state = False

        # display a message indicating recording has ended
        msg = QMessageBox(self)
        msg.setText(f"Recording Ended. Data recorded at<br>{os.path.abspath(self.camera.recording_path)}.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    @Slot(float)
    def check_recording_length(self, elapsed):
        '''
        When recording length has completed (if provided), end recording.
        '''
        if elapsed >= self.recording_length:
            self.on_stop_clicked()

    def closeEvent(self, event):
        '''
        When GUI is closed, stop streaming camera.
        '''
        self.camera.stop_camera()
        super().closeEvent(event)
