import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque

SECONDS_TO_CALIBRATE = 3.0
BLINK_THRESHOLD = 0.18
CONSECUTIVE_FRAMES_FOR_BLINK = 2
FACE_CONFIDENCE_LEVEL = 0.45
EYE_DARKNESS_THRESHOLD = 45.0
GAZE_SENSITIVITY = 0.07
AUDIO_SENSITIVITY = 2.0
SECONDS_FOR_EYES_CLOSED_WARNING = 3.0
SECONDS_TO_EXIT_IF_NO_FACE = 10.0

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_INDEX = 468
RIGHT_IRIS_INDEX = 473

current_audio_level = 0.0
audio_lock = threading.Lock()
audio_stream = None
baseline_audio_level = 0.000001

def audio_listener(indata, frames, time_info, status):
    global current_audio_level
    volume_per_channel = np.mean(indata, axis=1)
    volume = float(np.sqrt(np.mean(np.square(volume_per_channel))))
    with audio_lock:
        current_audio_level = volume

def start_microphone():
    global audio_stream
    try:
        audio_stream = sd.InputStream(callback=audio_listener, blocksize=1024, samplerate=22050, channels=1)
        audio_stream.start()
        return True
    except:
        print("Microphone not found. Running without audio.")
        return False

def calibrate_microphone():
    global baseline_audio_level
    try:
        print("Please stay quiet for 1 second...")
        recording = sd.rec(int(1.0 * 22050), samplerate=22050, channels=1, dtype='float64')
        sd.wait()
        volume_data = recording[:, 0]
        baseline_audio_level = float(np.sqrt(np.mean(np.square(volume_data))))
        print(f"Microphone calibrated.")
    except:
        baseline_audio_level = 0.000001

def get_eye_openness(landmarks, indices, width, height):
    try:
        points = [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]
        vertical_1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        vertical_2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    except:
        return 0.0

def get_eye_brightness(gray_frame, landmarks, indices, width, height):
    x_points = [int(landmarks[i].x * width) for i in indices]
    y_points = [int(landmarks[i].y * height) for i in indices]
    
    x_min = max(min(x_points) - 6, 0)
    x_max = min(max(x_points) + 6, width - 1)
    y_min = max(min(y_points) - 6, 0)
    y_max = min(max(y_points) + 6, height - 1)
    
    if x_max <= x_min or y_max <= y_min:
        return None
        
    eye_region = gray_frame[y_min:y_max, x_min:x_max]
    if eye_region.size == 0:
        return None
        
    return float(np.mean(eye_region))

def get_iris_center(landmarks):
    try:
        x = (landmarks[LEFT_IRIS_INDEX].x + landmarks[RIGHT_IRIS_INDEX].x) / 2.0
        y = (landmarks[LEFT_IRIS_INDEX].y + landmarks[RIGHT_IRIS_INDEX].y) / 2.0
        return x, y
    except:
        return None, None

mic_working = start_microphone()
if mic_working:
    calibrate_microphone()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Focus Monitor", cv2.WINDOW_NORMAL)
print("Look at the camera for 3 seconds to calibrate your eyes...")

calibration_x_values = []
calibration_y_values = []
start_time = time.time()

while time.time() - start_time < SECONDS_TO_CALIBRATE:
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        ix, iy = get_iris_center(landmarks)
        if ix is not None:
            calibration_x_values.append(ix)
            calibration_y_values.append(iy)
            
    cv2.putText(frame, "LOOK AT THE CAMERA...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Focus Monitor", frame)
    cv2.waitKey(1)

if not calibration_x_values:
    print("Failed to see face. Please try again with better light.")
    exit()

center_x_baseline = np.mean(calibration_x_values)
center_y_baseline = np.mean(calibration_y_values)
print("Calibration Done.")

score_history = deque(maxlen=6)
frame_counter = 0
blink_counter = 0
last_blink_timestamp = 0
eyes_closed_start_time = None
no_face_start_time = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_counter += 1
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detection_results = face_detection.process(rgb_frame)
    mesh_results = face_mesh.process(rgb_frame)
    
    face_confidence = 0.0
    if detection_results.detections:
        face_confidence = detection_results.detections[0].score[0]

    is_occluded = False
    did_blink = False
    looking_direction = "UNKNOWN"
    focus_score = 0
    is_noisy = False

    if mesh_results.multi_face_landmarks and face_confidence >= FACE_CONFIDENCE_LEVEL:
        no_face_start_time = None
        landmarks = mesh_results.multi_face_landmarks[0].landmark

        mp_drawing.draw_landmarks(frame, mesh_results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

        left_open = get_eye_openness(landmarks, LEFT_EYE_INDICES, width, height)
        right_open = get_eye_openness(landmarks, RIGHT_EYE_INDICES, width, height)
        avg_openness = (left_open + right_open) / 2.0

        if avg_openness > 0 and avg_openness < BLINK_THRESHOLD:
            blink_counter += 1
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            
            if time.time() - eyes_closed_start_time >= SECONDS_FOR_EYES_CLOSED_WARNING:
                cv2.putText(frame, "WAKE UP!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            if blink_counter >= CONSECUTIVE_FRAMES_FOR_BLINK:
                if time.time() - last_blink_timestamp > 0.35:
                    did_blink = True
                    last_blink_timestamp = time.time()
            blink_counter = 0
            eyes_closed_start_time = None

        left_bright = get_eye_brightness(gray_frame, landmarks, LEFT_EYE_INDICES, width, height)
        right_bright = get_eye_brightness(gray_frame, landmarks, RIGHT_EYE_INDICES, width, height)
        
        if left_bright is None or right_bright is None or left_bright < EYE_DARKNESS_THRESHOLD or right_bright < EYE_DARKNESS_THRESHOLD:
            is_occluded = True

        curr_x, curr_y = get_iris_center(landmarks)
        if curr_x is not None:
            diff_x = curr_x - center_x_baseline
            diff_y = curr_y - center_y_baseline
            
            if abs(diff_x) <= GAZE_SENSITIVITY and abs(diff_y) <= 0.06:
                looking_direction = "CENTER"
            elif abs(diff_x) > abs(diff_y):
                looking_direction = "LEFT" if diff_x < 0 else "RIGHT"
            else:
                looking_direction = "UP" if diff_y < 0 else "DOWN"

        head_x = landmarks[1].x
        head_y = landmarks[1].y
        is_head_straight = (abs(head_x - 0.5) < 0.22 and abs(head_y - 0.5) < 0.18)

        with audio_lock:
            live_volume = current_audio_level
        
        if baseline_audio_level > 0 and live_volume > baseline_audio_level * AUDIO_SENSITIVITY:
            is_noisy = True

        gaze_points = 1.0 if looking_direction == "CENTER" else 0.0
        head_points = 1.0 if is_head_straight else 0.0
        blink_penalty = 0.5 if did_blink else 0.0
        noise_penalty = 1.0 if is_noisy else 0.0
        
        raw_score = 0.4 * gaze_points + 0.3 * head_points + 0.2 * (1.0 - blink_penalty) + 0.1 * (1.0 - noise_penalty)
        if is_occluded:
            raw_score = raw_score * 0.2
            
        focus_score = int(raw_score * 100)

    else:
        focus_score = 0
        is_occluded = True
        looking_direction = "NO FACE"
        
        if no_face_start_time is None:
            no_face_start_time = time.time()
        else:
            seconds_gone = int(time.time() - no_face_start_time)
            cv2.putText(frame, f"Come back: {seconds_gone}s", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if seconds_gone >= SECONDS_TO_EXIT_IF_NO_FACE:
                print("You were gone too long. Bye!")
                break

    score_history.append(focus_score)
    final_score = int(np.mean(score_history)) if score_history else focus_score

    status_text = "FOCUSED"
    color = (0, 255, 0)
    
    if not mesh_results.multi_face_landmarks:
        status_text = "NO FACE"
        color = (0, 0, 255)
    elif is_noisy:
        status_text = "TOO LOUD"
        color = (0, 0, 255)
    elif did_blink:
        status_text = "BLINK"
        color = (255, 255, 0)
    elif final_score < 55:
        status_text = "DISTRACTED"
        color = (0, 165, 255)

    cv2.putText(frame, f"Score: {final_score}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Status: {status_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Looking: {looking_direction}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    bar_width = int((final_score / 100) * 300)
    cv2.rectangle(frame, (20, 160), (320, 190), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 160), (20 + bar_width, 190), color, -1)

    cv2.imshow("Focus Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if audio_stream:
    audio_stream.stop()
    audio_stream.close()
cap.release()
cv2.destroyAllWindows()