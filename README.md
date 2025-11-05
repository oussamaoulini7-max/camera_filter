# camera_filter

A real-time webcam filter application with AR-style overlays using Python, OpenCV, and MediaPipe.

## Features

- Apply fun filters like:
  - Cat ears and nose
  - Crown of roses
  - Dog ears
  - Rabbit ears and nose
  - Strange eyes
  - Sunglasses
- Real-time webcam face tracking using **MediaPipe Face Mesh**
- Simple GUI with **CustomTkinter**
- Switch between filters with buttons
- Start and stop camera anytime

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/camera_filter.git
cd camera_filter
``` 
2. Create and activate a virtual environment (Python 3.11 recommended):

```bash
# Create virtual environment
python -m venv venv

# Windows: activate the environment
venv\Scripts\activate

# macOS/Linux: activate the environment
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

requirements.txt should include:

```bash
opencv-python
mediapipe
customtkinter
Pillow
numpy
```

# Usage

```bash
python app.py
```

- Click Start Camera to begin.

- Select a filter button to apply it.

- Click Stop Camera to end the session.

# Notes

- Make sure all filter images are in the filters/ folder.

- Filters are PNG images with transparency.

- Designed for real-time webcam use.

- Works best with Python 3.11 and a properly activated virtual environment.

# License

- This project is licensed under the MIT License.
