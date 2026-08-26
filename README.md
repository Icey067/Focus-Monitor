# Focus Monitor

A real-time Focus Monitor built with **MediaPipe** and **OpenCV**. It intelligently tracks your face, eyes, head pose, and even background noise to estimate a real-time "focus level". It's a great tool for tracking your attention span while studying, working, or for building online-exam proctoring systems.

---

## 🎯 Features

*   **Real-time Facial Tracking:** Leverages MediaPipe Face Mesh for accurate, real-time facial landmark detection.
*   **Gaze & Head Pose Estimation:** Calculates whether you are looking center, left, right, up, or down.
*   **Audio Monitoring:** Uses `sounddevice` to calibrate background noise and detect loud noises that might indicate distraction.
*   **Eye Closure & Sleep Warning:** Alerts you with a "WAKE UP!" prompt if your eyes remain closed for more than 3 seconds.
*   **Absence Detection:** Automatically exits the program if no face is detected for an extended period (10 seconds).
*   **Dynamic Focus Score:** Calculates a 0-100% focus score based on head pose, gaze direction, blinking rates, and background noise.

## 🛠 Prerequisites

*   **Python 3.11** (Highly recommended due to `mediapipe` library compatibility).
*   A working **webcam**.
*   A working **microphone** (optional, but required for the noise detection feature).

## 🚀 Setup & Installation

> **Note:** The `mediapipe` library may have issues with newer Python versions (like 3.12+). To ensure this project runs smoothly, it is recommended to use a virtual environment based on **Python 3.11**.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/focus-monitor.git
cd focus-monitor
```

*(If you already have the files locally, just navigate to the project folder).*

### 2. Create a Virtual Environment

Navigate to this project's directory in your terminal and run the command that matches your Python 3.11 installation:

```bash
# On Linux/macOS (if 'python3.11' is available):
python3.11 -m venv .venv

# On Windows (if you used the official installer):
py -3.11 -m venv .venv

# If Python 3.11 is your default 'python3' command:
python3 -m venv .venv
```

### 3. Activate the Environment

```bash
# On Linux/macOS:
source .venv/bin/activate

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

With your virtual environment active, install the required packages:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

Run the main script to start the tracker:

```bash
python ml.py
```

### Calibration Phase
1.  **Audio Calibration:** Upon starting, the script will ask you to stay quiet for 1 second. This establishes a baseline for ambient noise.
2.  **Visual Calibration:** You will be prompted to look directly at the camera for 3 seconds. The program measures your baseline iris positions to calibrate gaze estimation.

### Monitoring Phase
*   The script will display a window with your webcam feed.
*   You will see an overlaid **Focus Score**, your current **Status** (e.g., FOCUSED, DISTRACTED, BLINK, NO FACE), and your **Looking Direction**.
*   To quit the application, press the **`q`** key on your keyboard while focused on the video window.

## ⚙️ Configuration

You can tweak the constants at the top of the `FocusMonitor` `__init__` method in `ml.py` to adjust sensitivities:

*   `SECONDS_TO_CALIBRATE`: Time given for eye calibration.
*   `BLINK_THRESHOLD`: Eye Aspect Ratio threshold to register a blink.
*   `GAZE_SENSITIVITY`: Sensitivity for determining if you are looking away from the center.
*   `AUDIO_SENSITIVITY`: Multiplier over the baseline audio level to trigger a "noise" penalty.
*   `SECONDS_FOR_EYES_CLOSED_WARNING`: How long eyes must be closed before the "WAKE UP!" text appears.

## 📝 License

See the [LICENSE](LICENSE) file for more information.
