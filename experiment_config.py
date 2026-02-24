import json
from pathlib import Path


DEFAULT_RECORDING_PARAMS = {
    'input_source': 'realsense',
    'source_path': None,
    'camera_index': 0,
    'serial_number': '213622074070',
    'folder_path': './data/',
    'recording_length': 10,
    'frames_per_file': 1000,
    'width': 640,
    'height': 480,
    'fps': 30,
    'codec': 'XVID',
    'exposure': 2500,
    'gain': 50,
    'laser_power': 200,
    'enable_ttl': True,
    'use_tracking': False,
    'ref_num_frames': 30,
    'tracking_method': 'dark',
    'use_window': True,
    'window_size': 100,
    'window_weight': 0.90,
    'loc_thresh': 99.5,
    'ksize': 5,
}


def load_recording_params(config_path=None, defaults=None):
    params = dict(DEFAULT_RECORDING_PARAMS)
    if defaults:
        params.update(defaults)

    if config_path is None:
        return params

    config_file = Path(config_path)
    if not config_file.exists():
        return params

    with config_file.open('r', encoding='utf-8') as file:
        loaded_params = json.load(file)

    if not isinstance(loaded_params, dict):
        raise ValueError(f'Config at {config_file} must be a JSON object.')

    params.update(loaded_params)
    return params