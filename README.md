## GUI to record infrared stream from RealSense D435f camera and save into avi files

This repository defines a `RealSenseCamera` class which engages a RealSense camera to grab <br>
frames and send output TTL signals on every frame. It also defines a `RealSenseGUI` class which <br>
creates a GUI written using `PySide6` to stream the video and to click record/stop recording. This system <br>
incorporates optional real-time tracking of the subject, using code adapted from [ezTrack](https://github.com/denisecailab/ezTrack).

### *Entry points*
- `run_recording.py`: RealSense recording GUI (no real-time tracking).
- `run_recording_tracking.py`: RealSense recording GUI with real-time tracking.
- `run_camera_preview.py`: Camera preview only (RealSense or USB via OpenCV), no recording GUI.

### *Logging*
- App-level logs are written to `./logs/app.log` (rotating file) and echoed to console.
- During each recording, a per-session log file is created at `<recording_path>/recording.log`.
- Logs include warnings/errors and key run parameters (camera ID, fps, resolution, codec, exposure/gain/laser, tracking flags).
***
### *To use the GUI:*
1. Navigate to this folder after cloning this repo: `cd /path/to/folder/RealSense_Recorder`.
2. Create conda environment: `conda env create -f your_env_file.yaml`.
    - Replace `your_env_file` with `env_mac` or `env_windows` depending on your OS.
3. Activate env: `source activate realsense`.
4. In the `run_recording.py` file, fill out the `recording_params` as desired.
    - Refer to the documentation in `gui.py` for details.
    - At runtime, connected cameras are discovered (RealSense + USB/OpenCV). If multiple are connected, you will be prompted to choose one.
5. Run `python run_recording.py` to boot up the GUI. Some notes:
    - A short camera preview opens first. Press `q` or `ESC` to close the preview and continue.
    - Recording GUI support is currently RealSense-only. If you select a USB camera, preview runs and the script exits.
    - _On Mac, you may need to run the above command using `sudo`._
    - _Sometimes, on boot you might get an error where the camera could not<br>connect. Try closing the GUI and running again._
    - _If recording from multiple streams simultaneously, it is recommended to<br>make multiple copies of the `run_recording.py` file, rename the files to<br>reflect the streams, and run them in their own separate terminal windows._
    - _It is also recommended to change the `folder_path` in each py file such that<br>the separate streams are saved to different folders._
6. Click 'Record' to start and 'Stop Recording' to stop.

***
### *To use with real-time tracking:*
1. Do steps 1-3 above.
2. In the `run_recording_tracking.py`, fill out the `recording_params` as desired.
    - The tracking parameters might need to be fine-tuned iteratively.
    - Real-time tracking is enabled as long as `use_tracking` is set to True in the `recording_params`.
3. Run `python run_recording_tracking.py` to boot up the GUI. See notes in above step 5.
    - Camera detection/selection and preview behavior are the same as in `run_recording.py` (including USB preview support and RealSense-only recording).
4. Click 'Compute Reference' to first create a reference image that will be used as a background for the real-time tracking.
    - This is done by taking a series of frames and taking the median projection, as in [ezTrack](https://github.com/denisecailab/ezTrack).
    - The number of frames is dictated by `ref_num_frames` in `recording_params`. Make sure the subject is sampling the space and not immobile while this reference is being computed.
    - If the reference creation is successful, a file called `reference.png` should be saved in your `folder_path`. You should now see live-tracking in the right panel of the GUI.
5. Click 'Record' to start recording of behavior and real-time tracking, and 'Stop Recording' to stop.

***
### *To preview cameras only (no recording GUI):*
1. Do steps 1-3 from the GUI section above.
2. (Optional) Edit `preview_params` in `run_camera_preview.py`.
    - `camera_id` can be a RealSense serial number or USB index string (e.g., `'0'`).
3. Run `python run_camera_preview.py`.
4. Select a camera if prompted. Press `q` or `ESC` to close preview.