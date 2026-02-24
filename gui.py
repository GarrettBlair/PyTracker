import os
import json
import csv
import cv2
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import ndimage
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from PySide6.QtCore import QThread, Signal, Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QHBoxLayout,
    QMessageBox,
    QInputDialog
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
            input_source='realsense',
            source_path=None,
            camera_index=0,
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
        self.input_source = input_source
        self.source_path = source_path
        self.camera_index = camera_index
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
        self.roi_rect = None
        self.target_zone = None
        self.depth_sensor = None
        self.pipeline = None
        self.config = None
        self.cap = None

        # set up source-specific stream config
        if self.input_source == 'realsense':
            if rs is None:
                raise ImportError("pyrealsense2 is required when input_source='realsense'.")
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
        elif self.input_source not in {'webcam', 'video_file'}:
            raise ValueError("Invalid 'input_source'. Must be one of ['realsense', 'webcam', 'video_file']")

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
                print(f"WARNING: ksize too large. Not applying for frame {self.total_frames}")
        
        com = ndimage.center_of_mass(diff) # returns as (y,x)

        return diff, com

    def run(self):
        '''
        Start camera and while recording is enabled (triggered
        externally), capture, display, and save frames. Display
        time elapsed as well.
        '''
        if self.input_source == 'realsense':
            profile = self.pipeline.start(self.config)
            device = profile.get_device()
            # note: depth and infrared are on the same sensor
            self.depth_sensor = device.first_depth_sensor()

            # manual exposure
            self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.depth_sensor.set_option(rs.option.exposure, self.exposure)
            self.depth_sensor.set_option(rs.option.gain, self.gain)
            self.depth_sensor.set_option(rs.option.laser_power, self.laser_power)
        elif self.input_source == 'webcam':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        elif self.input_source == 'video_file':
            if self.source_path is None:
                raise ValueError("'source_path' is required when input_source='video_file'.")
            self.cap = cv2.VideoCapture(self.source_path)

        if self.input_source in {'webcam', 'video_file'} and (self.cap is None or not self.cap.isOpened()):
            raise RuntimeError(f"Could not open source '{self.input_source}'.")

        try:
            while self.running:
                if self.input_source == 'realsense':
                    try:
                        frames = self.pipeline.wait_for_frames()
                    except RuntimeError:
                        print("Frame timeout. Retrying...")
                        continue

                    ir_frame = frames.get_infrared_frame()
                    if not ir_frame:
                        continue
                    frame_np = np.asanyarray(ir_frame.get_data())
                    frame_timestamp = ir_frame.get_timestamp()
                    hw_counter = -1
                    if ir_frame.supports_frame_metadata(rs.frame_metadata_value.frame_counter):
                        hw_counter = ir_frame.get_frame_metadata(rs.frame_metadata_value.frame_counter)
                else:
                    ret, frame_bgr = self.cap.read()
                    if not ret:
                        if self.input_source == 'video_file':
                            print('End of video file reached.')
                            break
                        continue
                    frame_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    frame_timestamp = time.time() * 1000.0
                    hw_counter = -1

                if self.computing_reference:
                    self.ref_stack.append(frame_np.copy())
                    if len(self.ref_stack) >= self.ref_num_frames:
                        self.reference = np.median(np.array(self.ref_stack), axis=0).astype(np.uint8)
                        self.computing_reference = False
                        self.reference_ready.emit(True)
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
                    with open(self.timestamps_csv_path, 'a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([self.total_frames-1, frame_timestamp, hw_counter]) # frame was already bumped in write_frame()

                else:
                    self.stats_data.emit(0, "00:00:00")

        finally:
            self.stop_recording()
            if self.pipeline is not None:
                self.pipeline.stop()
            if self.cap is not None:
                self.cap.release()

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
            
            # save params file of the recording's attributes
            params_keys = ['recording_start_time', 'recording_length', 'serial_number', 'folder_path', 'recording_path',
                        'frames_per_file', 'width', 'height', 'fps', 'codec', 'exposure', 'gain', 'laser_power', 'enable_ttl',
                        'input_source', 'source_path', 'camera_index', 'roi_rect', 'target_zone']
            params_info = {key: getattr(self, key) for key in params_keys}
            with open(os.path.join(self.recording_path, 'params.json'), 'w') as f:
                json.dump(params_info, f, indent=4)
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
                if self.depth_sensor is not None and self.depth_sensor.supports(rs.option.output_trigger_enabled):
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
            if self.depth_sensor is not None and self.depth_sensor.supports(rs.option.output_trigger_enabled):
                self.depth_sensor.set_option(rs.option.output_trigger_enabled, 0)

        if self.writer:
            self.writer.release()
            self.writer = None
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
                    print(f"Error processing timestamps: {e}")

        # compute distance traveled
        if self.use_tracking and self.tracking_csv_path:
            if os.path.exists(self.tracking_csv_path):
                try:
                    tracking_df = pd.read_csv(self.tracking_csv_path)
                    if len(tracking_df) > 0:
                        xy = tracking_df[['X','Y']].values
                        xy_diffs = xy[1:] - xy[:-1]
                        distances = np.nan_to_num(np.linalg.norm(xy_diffs, axis=1))
                        tracking_df['Distance_px'] = np.append(0, distances)
                        tracking_df.to_csv(self.tracking_csv_path, index=False)
                except Exception as e:
                    print(f"Error processing tracking data: {e}")

    def stop_camera(self):
        '''
        Stop running camera (used when GUI closes).
        '''
        self.running = False
        self.wait()


## Class to display and operate recording GUI ##
## ========================================== ##

class InteractiveVideoLabel(QLabel):
    mouse_pressed = Signal(int, int, object)
    mouse_moved = Signal(int, int)
    mouse_released = Signal(int, int, object)
    mouse_double_clicked = Signal(int, int, object)

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(int(event.position().x()), int(event.position().y()), event.button())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(int(event.position().x()), int(event.position().y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.mouse_released.emit(int(event.position().x()), int(event.position().y()), event.button())
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.mouse_double_clicked.emit(int(event.position().x()), int(event.position().y()), event.button())
        super().mouseDoubleClickEvent(event)

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
        self.recording_length = np.inf if (recording_length is None or recording_length <= 0) else recording_length
        self.frame_width = int(camera_kwargs.get('width', 640))
        self.frame_height = int(camera_kwargs.get('height', 480))
        self.roi_rect = None
        self.roi_start = None
        self.roi_end = None
        self.roi_dragging = False
        self.interaction_mode = None
        self.target_zone = None
        self.polygon_points = []
        self.last_frame_shape = (self.frame_height, self.frame_width)

        # ---------------------------------------------
        # UI layout initialization
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        # display stats
        self.stats_layout = QHBoxLayout()
        self.time_label = QLabel("Duration: 00:00:00")
        self.count_label = QLabel("Frames: 0")
        stat_font = QFont("Arial", 14, QFont.Bold)
        self.time_label.setFont(stat_font)
        self.count_label.setFont(stat_font)
        self.time_label.setStyleSheet("color: #2ecc71;")
        self.stats_layout.addWidget(self.time_label)
        self.stats_layout.addStretch()
        self.stats_layout.addWidget(self.count_label)
        self.layout.addLayout(self.stats_layout)

        self.video_container = QHBoxLayout()

        # display video label
        self.video_label = InteractiveVideoLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: black; border: 2px solid gray;"
            )
        self.video_label.setFixedSize(self.frame_width, self.frame_height)
        self.video_label.mouse_pressed.connect(self.on_video_mouse_pressed)
        self.video_label.mouse_moved.connect(self.on_video_mouse_moved)
        self.video_label.mouse_released.connect(self.on_video_mouse_released)
        self.video_label.mouse_double_clicked.connect(self.on_video_mouse_double_clicked)
        self.video_container.addWidget(self.video_label)

        # display tracking feed if using
        if self.use_tracking:
            self.tracking_label = InteractiveVideoLabel("Real-time Tracking")
            self.tracking_label.setAlignment(Qt.AlignCenter)
            self.tracking_label.setStyleSheet(
                "background-color: black; border: 2px solid red;"
            )
            self.tracking_label.setFixedSize(self.frame_width, self.frame_height)
            self.tracking_label.mouse_pressed.connect(self.on_video_mouse_pressed)
            self.tracking_label.mouse_moved.connect(self.on_video_mouse_moved)
            self.tracking_label.mouse_released.connect(self.on_video_mouse_released)
            self.tracking_label.mouse_double_clicked.connect(self.on_video_mouse_double_clicked)
            self.video_container.addWidget(self.tracking_label)
        
        self.layout.addLayout(self.video_container)

        # create and connect buttons
        self.btn_layout = QHBoxLayout()

        if self.use_tracking:
            self.reference_btn = QPushButton("Compute Reference")
            self.reference_btn.clicked.connect(self.on_reference_clicked)
            self.btn_layout.addWidget(self.reference_btn)

            self.roi_btn = QPushButton("Set ROI")
            self.roi_btn.clicked.connect(self.on_set_roi_clicked)
            self.clear_roi_btn = QPushButton("Clear ROI")
            self.clear_roi_btn.clicked.connect(self.on_clear_roi_clicked)
            self.arc_btn = QPushButton("Set Arc")
            self.arc_btn.clicked.connect(self.on_set_arc_clicked)
            self.circle_btn = QPushButton("Set Circle")
            self.circle_btn.clicked.connect(self.on_set_circle_clicked)
            self.polygon_btn = QPushButton("Set Polygon")
            self.polygon_btn.clicked.connect(self.on_set_polygon_clicked)
            self.clear_target_btn = QPushButton("Clear Target")
            self.clear_target_btn.clicked.connect(self.on_clear_target_clicked)

            self.btn_layout.addWidget(self.roi_btn)
            self.btn_layout.addWidget(self.clear_roi_btn)
            self.btn_layout.addWidget(self.arc_btn)
            self.btn_layout.addWidget(self.circle_btn)
            self.btn_layout.addWidget(self.polygon_btn)
            self.btn_layout.addWidget(self.clear_target_btn)
        
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
        self.last_frame_shape = (height, width)

        overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self.draw_roi_overlay(overlay)
        self.draw_target_overlay(overlay)

        bytes_per_line = width * 3
        q_img = QImage(overlay.data, width, height, bytes_per_line, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))
    
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
        diff_max = diff.max()
        diff_min = diff.min()
        if diff_max == diff_min:
            diff = np.zeros(diff.shape, dtype='uint8')
        else:
            diff = ((diff - diff_min) / (diff_max - diff_min) * 255).astype('uint8')
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

    def on_set_roi_clicked(self):
        self.interaction_mode = 'roi'
        self.roi_start = None
        self.roi_end = None
        self.roi_dragging = False

    def on_clear_roi_clicked(self):
        self.roi_rect = None
        self.target_zone = None
        self.polygon_points = []
        self.roi_start = None
        self.roi_end = None
        self.roi_dragging = False
        self.interaction_mode = None

    def on_set_arc_clicked(self):
        if self.roi_rect is None:
            QMessageBox.warning(self, "ROI Required", "Set ROI first.")
            return

        min_radius, ok = QInputDialog.getDouble(self, "Arc Target", "Minimum Radius [%]:", 20.0, 0.0, 100.0, 1)
        if not ok:
            return
        max_radius, ok = QInputDialog.getDouble(self, "Arc Target", "Maximum Radius [%]:", 50.0, 0.0, 100.0, 1)
        if not ok:
            return
        place_angle, ok = QInputDialog.getDouble(self, "Arc Target", "Place Angle [deg]:", 0.0, -360.0, 360.0, 1)
        if not ok:
            return
        angle_width, ok = QInputDialog.getDouble(self, "Arc Target", "Angle Width [deg]:", 60.0, 0.1, 360.0, 1)
        if not ok:
            return

        if max_radius < min_radius:
            QMessageBox.warning(self, "Invalid Arc", "Maximum Radius must be >= Minimum Radius.")
            return

        self.target_zone = {
            'type': 'arc',
            'min_radius_pct': float(min_radius),
            'max_radius_pct': float(max_radius),
            'place_angle_deg': float(place_angle),
            'angle_width_deg': float(angle_width),
        }
        self.interaction_mode = None

    def on_set_circle_clicked(self):
        if self.roi_rect is None:
            QMessageBox.warning(self, "ROI Required", "Set ROI first.")
            return

        place_angle, ok = QInputDialog.getDouble(self, "Circle Target", "Place Angle [deg]:", 0.0, -360.0, 360.0, 1)
        if not ok:
            return
        place_radius, ok = QInputDialog.getDouble(self, "Circle Target", "Place Radius [%]:", 40.0, 0.0, 100.0, 1)
        if not ok:
            return
        target_radius, ok = QInputDialog.getDouble(self, "Circle Target", "Target Radius [%]:", 15.0, 0.1, 100.0, 1)
        if not ok:
            return

        self.target_zone = {
            'type': 'circle',
            'place_angle_deg': float(place_angle),
            'place_radius_pct': float(place_radius),
            'target_radius_pct': float(target_radius),
        }
        self.interaction_mode = None

    def on_set_polygon_clicked(self):
        if self.roi_rect is None:
            QMessageBox.warning(self, "ROI Required", "Set ROI first.")
            return
        self.polygon_points = []
        self.interaction_mode = 'polygon'

    def on_clear_target_clicked(self):
        self.target_zone = None
        self.polygon_points = []
        if self.interaction_mode == 'polygon':
            self.interaction_mode = None

    @Slot(int, int, object)
    def on_video_mouse_pressed(self, x, y, button):
        frame_point = self.display_to_frame_coords(x, y)
        if frame_point is None:
            return

        fx, fy = frame_point
        if self.interaction_mode == 'roi':
            self.roi_start = (fx, fy)
            self.roi_end = (fx, fy)
            self.roi_dragging = True
        elif self.interaction_mode == 'polygon' and button == Qt.MouseButton.LeftButton:
            if self.point_in_roi(fx, fy):
                self.polygon_points.append((fx, fy))
        elif self.interaction_mode == 'polygon' and button == Qt.MouseButton.RightButton:
            if self.polygon_points:
                self.polygon_points.pop()

    @Slot(int, int)
    def on_video_mouse_moved(self, x, y):
        frame_point = self.display_to_frame_coords(x, y)
        if frame_point is None:
            return
        fx, fy = frame_point

        if self.interaction_mode == 'roi' and self.roi_dragging and self.roi_start is not None:
            if self.is_shift_pressed():
                self.roi_end = self.constrain_square_point(self.roi_start, (fx, fy))
            else:
                self.roi_end = (fx, fy)

    @Slot(int, int, object)
    def on_video_mouse_released(self, x, y, button):
        frame_point = self.display_to_frame_coords(x, y)
        if frame_point is None:
            return
        fx, fy = frame_point

        if self.interaction_mode == 'roi' and self.roi_dragging and self.roi_start is not None:
            if self.is_shift_pressed():
                self.roi_end = self.constrain_square_point(self.roi_start, (fx, fy))
            else:
                self.roi_end = (fx, fy)
            self.roi_rect = self.make_rect(self.roi_start, self.roi_end)
            self.roi_start = None
            self.roi_end = None
            self.roi_dragging = False
            self.interaction_mode = None

    @Slot(int, int, object)
    def on_video_mouse_double_clicked(self, x, y, button):
        if self.interaction_mode == 'polygon' and button == Qt.MouseButton.LeftButton and len(self.polygon_points) >= 3:
            self.target_zone = {
                'type': 'polygon',
                'points': [[int(px), int(py)] for px, py in self.polygon_points],
            }
            self.interaction_mode = None

    def display_to_frame_coords(self, x, y):
        if self.last_frame_shape is None:
            return None

        frame_h, frame_w = self.last_frame_shape
        if frame_w <= 0 or frame_h <= 0:
            return None

        label_w = max(1, self.video_label.width())
        label_h = max(1, self.video_label.height())

        fx = int(np.clip((x / label_w) * frame_w, 0, frame_w - 1))
        fy = int(np.clip((y / label_h) * frame_h, 0, frame_h - 1))
        return fx, fy

    def make_rect(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        x_min = int(min(x1, x2))
        y_min = int(min(y1, y2))
        width = int(abs(x2 - x1))
        height = int(abs(y2 - y1))
        if width < 2 or height < 2:
            return None
        return (x_min, y_min, width, height)

    def is_shift_pressed(self):
        modifiers = QApplication.keyboardModifiers()
        return bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

    def constrain_square_point(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0

        side = max(abs(dx), abs(dy))
        if side == 0:
            return end

        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1

        frame_h, frame_w = self.last_frame_shape
        square_x = int(np.clip(x0 + sx * side, 0, frame_w - 1))
        square_y = int(np.clip(y0 + sy * side, 0, frame_h - 1))
        return square_x, square_y

    def point_in_roi(self, x, y):
        if self.roi_rect is None:
            return False
        rx, ry, rw, rh = self.roi_rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

    def draw_roi_overlay(self, frame_bgr):
        if self.roi_rect is not None:
            rx, ry, rw, rh = self.roi_rect
            cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

            # draw 4x4 guide grid inside ROI (3 interior vertical + 3 interior horizontal lines)
            for i in range(1, 4):
                xg = int(rx + (rw * i / 4.0))
                yg = int(ry + (rh * i / 4.0))
                cv2.line(frame_bgr, (xg, ry), (xg, ry + rh), (0, 180, 0), 1)
                cv2.line(frame_bgr, (rx, yg), (rx + rw, yg), (0, 180, 0), 1)
        elif self.interaction_mode == 'roi' and self.roi_start is not None and self.roi_end is not None:
            rect = self.make_rect(self.roi_start, self.roi_end)
            if rect is not None:
                rx, ry, rw, rh = rect
                cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    def draw_target_overlay(self, frame_bgr):
        if self.roi_rect is None:
            return

        if self.interaction_mode == 'polygon' and len(self.polygon_points) > 0:
            points_np = np.array(self.polygon_points, dtype=np.int32)
            cv2.polylines(frame_bgr, [points_np], False, (255, 0, 255), 2)

        if self.target_zone is None:
            return

        target_type = self.target_zone.get('type')
        rx, ry, rw, rh = self.roi_rect
        cx = int(rx + rw / 2)
        cy = int(ry + rh / 2)
        roi_radius = max(1.0, min(rw, rh) / 2.0)
        roi_radius_x = max(1.0, rw / 2.0)
        roi_radius_y = max(1.0, rh / 2.0)

        if target_type == 'arc':
            min_rx = max(1, int(roi_radius_x * self.target_zone['min_radius_pct'] / 100.0))
            min_ry = max(1, int(roi_radius_y * self.target_zone['min_radius_pct'] / 100.0))
            max_rx = max(min_rx, int(roi_radius_x * self.target_zone['max_radius_pct'] / 100.0))
            max_ry = max(min_ry, int(roi_radius_y * self.target_zone['max_radius_pct'] / 100.0))
            angle = float(self.target_zone['place_angle_deg'])
            width = float(self.target_zone['angle_width_deg'])
            start_angle = angle - width / 2.0
            end_angle = angle + width / 2.0

            cv2.ellipse(frame_bgr, (cx, cy), (max_rx, max_ry), 0, start_angle, end_angle, (255, 0, 0), 2)
            cv2.ellipse(frame_bgr, (cx, cy), (min_rx, min_ry), 0, start_angle, end_angle, (255, 0, 0), 2)

            start_rad = np.deg2rad(start_angle)
            end_rad = np.deg2rad(end_angle)
            p1_inner = (int(cx + min_rx * np.cos(start_rad)), int(cy + min_ry * np.sin(start_rad)))
            p1_outer = (int(cx + max_rx * np.cos(start_rad)), int(cy + max_ry * np.sin(start_rad)))
            p2_inner = (int(cx + min_rx * np.cos(end_rad)), int(cy + min_ry * np.sin(end_rad)))
            p2_outer = (int(cx + max_rx * np.cos(end_rad)), int(cy + max_ry * np.sin(end_rad)))
            cv2.line(frame_bgr, p1_inner, p1_outer, (255, 0, 0), 2)
            cv2.line(frame_bgr, p2_inner, p2_outer, (255, 0, 0), 2)

        elif target_type == 'circle':
            angle = np.deg2rad(float(self.target_zone['place_angle_deg']))
            place_radius = float(self.target_zone['place_radius_pct'])
            target_radius = float(self.target_zone['target_radius_pct'])

            center_dist = roi_radius * place_radius / 100.0
            target_center = (
                int(cx + center_dist * np.cos(angle)),
                int(cy + center_dist * np.sin(angle))
            )
            target_r = max(1, int(roi_radius * target_radius / 100.0))
            cv2.circle(frame_bgr, target_center, target_r, (0, 255, 255), 2)

        elif target_type == 'polygon':
            points = self.target_zone.get('points', [])
            if len(points) >= 3:
                points_np = np.array(points, dtype=np.int32)
                cv2.polylines(frame_bgr, [points_np], True, (255, 0, 255), 2)
    
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

        self.camera.roi_rect = self.roi_rect
        self.camera.target_zone = self.target_zone

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
