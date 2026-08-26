"""
Focus Monitor - Real-time focus tracking using MediaPipe and OpenCV.
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class FocusMonitor:
    def __init__(self):
        # Configuration / Thresholds
        self.SECONDS_TO_CALIBRATE = 3.0
        self.BLINK_THRESHOLD = 0.18
        self.CONSECUTIVE_FRAMES_FOR_BLINK = 2
        self.FACE_CONFIDENCE_LEVEL = 0.45
        self.EYE_DARKNESS_THRESHOLD = 45.0
        self.GAZE_SENSITIVITY = 0.07
        self.AUDIO_SENSITIVITY = 2.0
        self.SECONDS_FOR_EYES_CLOSED_WARNING = 3.0
        self.SECONDS_TO_EXIT_IF_NO_FACE = 10.0

        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.4)

        # Facial landmarks indices
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS_INDEX = 468
        self.RIGHT_IRIS_INDEX = 473

        # Audio state
        self.current_audio_level = 0.0
        self.audio_lock = threading.Lock()
        self.audio_stream = None
        self.baseline_audio_level = 0.000001
        self.mic_working = False

        # State tracking
        self.center_x_baseline = None
        self.center_y_baseline = None
        self.score_history = deque(maxlen=6)
        self.frame_counter = 0
        self.blink_counter = 0
        self.last_blink_timestamp = 0
        self.eyes_closed_start_time = None
        self.no_face_start_time = None

    def audio_listener(self, indata, frames, time_info, status):
        """Callback to process audio stream and calculate volume level."""
        volume_per_channel = np.mean(indata, axis=1)
        volume = float(np.sqrt(np.mean(np.square(volume_per_channel))))
        with self.audio_lock:
            self.current_audio_level = volume

    def start_microphone(self):
        """Initializes and starts the microphone stream."""
        try:
            self.audio_stream = sd.InputStream(callback=self.audio_listener, blocksize=1024, samplerate=22050, channels=1)
            self.audio_stream.start()
            self.mic_working = True
            logging.info("Microphone initialized successfully.")
            return True
        except Exception as e:
            logging.warning(f"Microphone not found or could not be initialized: {e}. Running without audio.")
            self.mic_working = False
            return False

    def calibrate_microphone(self):
        """Calibrates baseline audio level for noise detection."""
        try:
            logging.info("Please stay quiet for 1 second to calibrate microphone...")
            recording = sd.rec(int(1.0 * 22050), samplerate=22050, channels=1, dtype='float64')
            sd.wait()
            volume_data = recording[:, 0]
            self.baseline_audio_level = float(np.sqrt(np.mean(np.square(volume_data))))
            logging.info(f"Microphone calibrated. Baseline level: {self.baseline_audio_level:.6f}")
        except Exception as e:
            logging.error(f"Failed to calibrate microphone: {e}")
            self.baseline_audio_level = 0.000001

    def get_eye_openness(self, landmarks, indices, width, height):
        """Calculates Eye Aspect Ratio (EAR) for given eye indices."""
        try:
            points = [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]
            vertical_1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            vertical_2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
            horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
            if horizontal == 0:
                return 0.0
            return (vertical_1 + vertical_2) / (2.0 * horizontal)
        except Exception:
            return 0.0

    def get_eye_brightness(self, gray_frame, landmarks, indices, width, height):
        """Calculates average brightness of the eye region to detect occlusions."""
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

    def get_iris_center(self, landmarks):
        """Finds the center point between both irises."""
        try:
            x = (landmarks[self.LEFT_IRIS_INDEX].x + landmarks[self.RIGHT_IRIS_INDEX].x) / 2.0
            y = (landmarks[self.LEFT_IRIS_INDEX].y + landmarks[self.RIGHT_IRIS_INDEX].y) / 2.0
            return x, y
        except Exception:
            return None, None

    def calibrate_eyes(self, cap):
        """Calibrates initial gaze position."""
        cv2.namedWindow("Focus Monitor", cv2.WINDOW_NORMAL)
        logging.info(f"Look at the camera for {self.SECONDS_TO_CALIBRATE} seconds to calibrate your eyes...")
        
        calibration_x_values = []
        calibration_y_values = []
        start_time = time.time()
        
        while time.time() - start_time < self.SECONDS_TO_CALIBRATE:
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ix, iy = self.get_iris_center(landmarks)
                if ix is not None:
                    calibration_x_values.append(ix)
                    calibration_y_values.append(iy)
                    
            cv2.putText(frame, "LOOK AT THE CAMERA...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Focus Monitor", frame)
            cv2.waitKey(1)
        
        if not calibration_x_values:
            logging.error("Failed to see face. Please try again with better light.")
            return False
        
        self.center_x_baseline = np.mean(calibration_x_values)
        self.center_y_baseline = np.mean(calibration_y_values)
        logging.info("Eye calibration done.")
        return True

    def process_frame(self, frame):
        """Processes a single video frame to calculate focus score."""
        self.frame_counter += 1
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detection_results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)
        
        face_confidence = 0.0
        if detection_results.detections:
            face_confidence = detection_results.detections[0].score[0]

        is_occluded = False
        did_blink = False
        looking_direction = "UNKNOWN"
        focus_score = 0
        is_noisy = False

        if mesh_results.multi_face_landmarks and face_confidence >= self.FACE_CONFIDENCE_LEVEL:
            self.no_face_start_time = None
            landmarks = mesh_results.multi_face_landmarks[0].landmark

            self.mp_drawing.draw_landmarks(frame, mesh_results.multi_face_landmarks[0], self.mp_face_mesh.FACEMESH_TESSELATION,
                                      self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

            left_open = self.get_eye_openness(landmarks, self.LEFT_EYE_INDICES, width, height)
            right_open = self.get_eye_openness(landmarks, self.RIGHT_EYE_INDICES, width, height)
            avg_openness = (left_open + right_open) / 2.0

            if avg_openness > 0 and avg_openness < self.BLINK_THRESHOLD:
                self.blink_counter += 1
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()
                
                if time.time() - self.eyes_closed_start_time >= self.SECONDS_FOR_EYES_CLOSED_WARNING:
                    cv2.putText(frame, "WAKE UP!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                if self.blink_counter >= self.CONSECUTIVE_FRAMES_FOR_BLINK:
                    if time.time() - self.last_blink_timestamp > 0.35:
                        did_blink = True
                        self.last_blink_timestamp = time.time()
                self.blink_counter = 0
                self.eyes_closed_start_time = None

            left_bright = self.get_eye_brightness(gray_frame, landmarks, self.LEFT_EYE_INDICES, width, height)
            right_bright = self.get_eye_brightness(gray_frame, landmarks, self.RIGHT_EYE_INDICES, width, height)
            
            if left_bright is None or right_bright is None or left_bright < self.EYE_DARKNESS_THRESHOLD or right_bright < self.EYE_DARKNESS_THRESHOLD:
                is_occluded = True

            curr_x, curr_y = self.get_iris_center(landmarks)
            if curr_x is not None:
                diff_x = curr_x - self.center_x_baseline
                diff_y = curr_y - self.center_y_baseline
                
                if abs(diff_x) <= self.GAZE_SENSITIVITY and abs(diff_y) <= 0.06:
                    looking_direction = "CENTER"
                elif abs(diff_x) > abs(diff_y):
                    looking_direction = "LEFT" if diff_x < 0 else "RIGHT"
                else:
                    looking_direction = "UP" if diff_y < 0 else "DOWN"

            head_x = landmarks[1].x
            head_y = landmarks[1].y
            is_head_straight = (abs(head_x - 0.5) < 0.22 and abs(head_y - 0.5) < 0.18)

            with self.audio_lock:
                live_volume = self.current_audio_level
            
            if self.baseline_audio_level > 0 and live_volume > self.baseline_audio_level * self.AUDIO_SENSITIVITY:
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
            
            if self.no_face_start_time is None:
                self.no_face_start_time = time.time()
            else:
                seconds_gone = int(time.time() - self.no_face_start_time)
                cv2.putText(frame, f"Come back: {seconds_gone}s", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if seconds_gone >= self.SECONDS_TO_EXIT_IF_NO_FACE:
                    logging.info("You were gone too long. Exiting.")
                    return frame, False # Signal to exit

        self.score_history.append(focus_score)
        final_score = int(np.mean(self.score_history)) if self.score_history else focus_score

        # Drawing the UI overlay
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
        
        return frame, True

    def run(self):
        """Starts the focus monitoring loop."""
        if self.start_microphone():
            self.calibrate_microphone()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Could not open camera.")
            return

        if not self.calibrate_eyes(cap):
            cap.release()
            cv2.destroyAllWindows()
            return

        logging.info("Starting Focus Monitor...")
        while True:
            success, frame = cap.read()
            if not success:
                logging.error("Failed to read frame from camera.")
                break

            processed_frame, should_continue = self.process_frame(frame)
            cv2.imshow("Focus Monitor", processed_frame)
            
            if not should_continue or (cv2.waitKey(1) & 0xFF == ord('q')):
                break

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Focus Monitor stopped.")

if __name__ == "__main__":
    monitor = FocusMonitor()
    monitor.run()