## GUI to record infrared stream from RealSense D435f camera and save into avi files

This repository defines a `RealSenseCamera` class which engages a RealSense camera to grab <br>
frames and send output TTL signals on every frame. It also defines a `RealSenseGUI` class which <br>
creates a GUI written using `PySide6` to stream the video and to click record/stop recording. This system <br>
incorporates optional real-time tracking of the subject, using code adapted from [ezTrack](https://github.com/denisecailab/ezTrack).
***
### *To use the GUI:*
1. Navigate to this folder after cloning this repo: `cd /path/to/folder/RealSense_Recorder`.
2. Create conda environment: `conda env create -f your_env_file.yaml`.
    - Replace `your_env_file` with `env_mac` or `env_windows` depending on your OS.
3. Activate env: `source activate realsense`.
4. Edit `recording_config.json` to set your experiment parameters.
    - `input_source` can be one of: `realsense`, `webcam`, or `video_file`.
    - For `webcam`, use `camera_index`.
    - For `video_file`, set `source_path` to your video path.
    - Refer to documentation in `gui.py` for parameter details.
5. Run `python run_recording.py` to boot up the GUI. Some notes:
    - _On Mac, you may need to run the above command using `sudo`._
    - _Sometimes, on boot you might get an error where the camera could not<br>connect. Try closing the GUI and running again._
    - _If recording from multiple streams simultaneously, it is recommended to<br>make multiple copies of the `run_recording.py` file, rename the files to<br>reflect the streams, and run them in their own separate terminal windows._
    - _It is also recommended to change the `folder_path` in each py file such that<br>the separate streams are saved to different folders._
6. Click 'Record' to start and 'Stop Recording' to stop.

***
### *To use with real-time tracking:*
1. Do steps 1-3 above.
2. In `recording_config.json`, set `use_tracking` to `true` and adjust tracking parameters as desired.
    - The tracking parameters might need to be fine-tuned iteratively.
    - Real-time tracking is enabled as long as `use_tracking` is set to True in the `recording_params`.
3. Run `python run_recording_tracking.py` to boot up the GUI. See notes in above step 5.
4. Click 'Compute Reference' to first create a reference image that will be used as a background for the real-time tracking.
    - This is done by taking a series of frames and taking the median projection, as in [ezTrack](https://github.com/denisecailab/ezTrack).
    - The number of frames is dictated by `ref_num_frames` in `recording_params`. Make sure the subject is sampling the space and not immobile while this reference is being computed.
    - If the reference creation is successful, a file called `reference.png` should be saved in your `folder_path`. You should now see live-tracking in the right panel of the GUI.
5. Define ROI and target zone:
    - Click `Set ROI`, then click-drag on the left video panel to draw a bounding box.
    - Click `Set Arc`, `Set Circle`, or `Set Polygon` to define the target zone within the ROI.
    - For polygon targets, left-click to add vertices, right-click to remove the last point, and double-left-click to finalize.
    - Use `Clear ROI` to reset both ROI and target, or `Clear Target` to keep ROI and remove only target.
6. Click 'Record' to start recording of behavior and real-time tracking, and 'Stop Recording' to stop.