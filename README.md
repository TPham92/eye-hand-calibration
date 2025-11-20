# Hand-Eye Calibration Tool

A GUI-based hand-eye calibration utility using:

- Intel RealSense camera
- OpenCV ArUco / ChArUco board
- Robot-reported base-frame coordinates

The tool collects corresponding 3D camera points and robot points, then computes a rigid transform suitable for robotic manipulation and vision-based tasks.


## Installation

Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

## Generate ChArUro board
```bash
python .\src\generate_charuco_board.py
```

## Eye to hand calibration

```bash
python .\src\eye_hand_calibration.py
```

## Configuration

All settings come from config.json:

```json
{
  "charuco_marker":{
    "dictionary": "DICT_4X4_50",
    "rows": 5,
    "columns": 7,
    "square_size": 39.0,
    "marker_size": 31.0,
    "render_dpi": 300,
    "margin": 2.0
  },
  "rs_camera": {
    "width": 1280,
    "height": 720,
    "fps": 30
  }
}
```


