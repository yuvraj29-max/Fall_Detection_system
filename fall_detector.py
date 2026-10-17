"""
AI Fall Detection System (Webcam & ESP32-CAM Hybrid)
---------------------------------------------------
Monitors a person's posture in real-time using YOLOv8-pose.
Classifies states: STANDING, SITTING, LYING / FALL RISK, and FALL CONFIRMED.
Includes audio alerts, grace period countdown, false-alarm cancel key,
and full bidirectional integration with ESP32-CAM hardware (Buzzer, LEDs, MPU6050, PIR, Button).
"""

import os
import sys
import time
import math
import json
import threading
import requests
import cv2
import numpy as np

# Ensure unbuffered terminal output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Audio support (Windows winsound / pyttsx3)
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 160)
    HAS_TTS = True
except Exception:
    HAS_TTS = False

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not installed. Please install via: pip install ultralytics")
    sys.exit(1)

from notifier import NotificationManager


class AudioAlert:
    """Manages non-blocking audio alerts so video stream does not stutter."""
    def __init__(self):
        self._last_speech_time = 0
        self._is_speaking = False

    def beep(self, frequency=1000, duration_ms=250):
        if HAS_WINSOUND:
            threading.Thread(
                target=lambda: winsound.Beep(frequency, duration_ms),
                daemon=True
            ).start()

    def speak(self, text, min_interval=4.0):
        now = time.time()
        if HAS_TTS and not self._is_speaking and (now - self._last_speech_time > min_interval):
            self._last_speech_time = now
            def _say():
                self._is_speaking = True
                try:
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
                finally:
                    self._is_speaking = False
            threading.Thread(target=_say, daemon=True).start()


class ThreadedCamera:
    """Eliminates network stream buffer latency by constantly pulling the latest frame in a background thread."""
    def __init__(self, src, is_stream=True):
        self.is_stream = is_stream
        if is_stream:
            self.cap = cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)

        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()


class ESP32Bridge:
    """Handles communication with the ESP32-CAM hardware peripherals."""
    def __init__(self, ip=None, on_button_press=None):
        self.ip = ip
        self.on_button_press = on_button_press
        self.connected = False
        self.last_status = {}
        self._running = True

        if self.ip:
            threading.Thread(target=self._poll_loop, daemon=True).start()

    def set_alert(self, active=True):
        """Sends command to ESP32 to trigger or silence Buzzer & LEDs."""
        if not self.ip:
            return
        def _send():
            try:
                state_val = "1" if active else "0"
                requests.get(f"http://{self.ip}/alert?state={state_val}", timeout=1.5)
            except Exception:
                pass
        threading.Thread(target=_send, daemon=True).start()

    def _poll_loop(self):
        """Continuously reads sensor telemetry and hardware button state from ESP32."""
        while self._running:
            try:
                r = requests.get(f"http://{self.ip}/status", timeout=1.2)
                if r.status_code == 200:
                    data = r.json()
                    self.last_status = data
                    self.connected = True
                    # Check if physical cancel button on ESP32 was pressed
                    if data.get("button_pressed", 0) == 1:
                        if self.on_button_press:
                            self.on_button_press()
            except Exception:
                self.connected = False
            time.sleep(0.25)


class FallDetector:
    # Keypoint indices from COCO format used by YOLOv8-pose:
    # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
    # 5: L-Shoulder, 6: R-Shoulder
    # 7: L-Elbow, 8: R-Elbow
    # 9: L-Wrist, 10: R-Wrist
    # 11: L-Hip, 12: R-Hip
    # 13: L-Knee, 14: R-Knee
    # 15: L-Ankle, 16: R-Ankle

    def __init__(
        self,
        model_name="yolov8n-pose.pt",
        fall_confirm_seconds=2.5,
        emergency_dispatch_seconds=10.0,
        torso_angle_threshold=42.0,
        aspect_ratio_threshold=0.95
    ):
        print(f"[INFO] Loading pose estimation model: {model_name}...")
        self.model = YOLO(model_name)
        print("[INFO] Model loaded successfully.")

        self.fall_confirm_seconds = fall_confirm_seconds
        self.emergency_dispatch_seconds = emergency_dispatch_seconds
        self.torso_angle_threshold = torso_angle_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold

        self.audio = AudioAlert()
        self.notifier = NotificationManager()
        self.esp32 = None  # Set dynamically in main()
        self.current_frame = None
        self.last_pose_details = {}

        # State tracking
        self.state = "STAND"
        self.fall_start_time = None
        self.confirmed_fall_time = None
        self.emergency_dispatched = False
        self.is_alarm_active = False

        # History for velocity calculation
        self.hip_y_history = []
        self.max_history = 10

    def calculate_angle(self, p1, p2):
        """Calculates absolute inclination angle in degrees relative to the horizontal plane."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        rad = math.atan2(abs(dy), abs(dx))
        return math.degrees(rad)

    def analyze_pose(self, keypoints, bbox):
        """
        keypoints: (17, 3) [x, y, conf]
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        aspect_ratio = width / height

        # Get key joints
        l_sh = keypoints[5][:2]
        r_sh = keypoints[6][:2]
        l_hip = keypoints[11][:2]
        r_hip = keypoints[12][:2]
        l_knee = keypoints[13][:2]
        r_knee = keypoints[14][:2]
        l_ankle = keypoints[15][:2]
        r_ankle = keypoints[16][:2]

        conf_sh = (keypoints[5][2] + keypoints[6][2]) / 2.0
        conf_hip = (keypoints[11][2] + keypoints[12][2]) / 2.0

        # Midpoints
        mid_shoulder = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        mid_knee = ((l_knee[0] + r_knee[0]) / 2.0, (l_knee[1] + r_knee[1]) / 2.0)
        mid_ankle = ((l_ankle[0] + r_ankle[0]) / 2.0, (l_ankle[1] + r_ankle[1]) / 2.0)

        # Torso inclination angle
        if conf_sh > 0.3 and conf_hip > 0.3:
            torso_angle = self.calculate_angle(mid_hip, mid_shoulder)
        else:
            torso_angle = 90.0 if aspect_ratio < 0.9 else 20.0

        # Rapid downward descent tracking
        self.hip_y_history.append((time.time(), mid_hip[1]))
        if len(self.hip_y_history) > self.max_history:
            self.hip_y_history.pop(0)

        rapid_drop = False
        if len(self.hip_y_history) >= 4:
            dt = self.hip_y_history[-1][0] - self.hip_y_history[0][0]
            dy = self.hip_y_history[-1][1] - self.hip_y_history[0][1]
            if dt > 0:
                velocity = dy / dt
                if velocity > height * 1.5:
                    rapid_drop = True

        is_horizontal = (torso_angle < self.torso_angle_threshold)
        is_wide_bbox = (aspect_ratio > self.aspect_ratio_threshold)
        is_upright = torso_angle >= 55.0
        thigh_span = abs(mid_hip[1] - mid_knee[1])

        detected_posture = "UNKNOWN"

        if is_horizontal or (is_wide_bbox and torso_angle < 50.0):
            detected_posture = "LYING / FALLEN"
        elif is_upright:
            if aspect_ratio > 0.65 and thigh_span < height * 0.28:
                detected_posture = "SITTING"
            else:
                detected_posture = "STANDING"
        else:
            if aspect_ratio > 0.85:
                detected_posture = "LYING / FALLEN"
            else:
                detected_posture = "SITTING"

        return detected_posture, torso_angle, aspect_ratio, rapid_drop

    def update(self, detected_posture, torso_angle, aspect_ratio, frame=None):
        current_time = time.time()
        self.last_pose_details = {
            "angle": torso_angle,
            "aspect_ratio": aspect_ratio,
            "posture": detected_posture
        }
        if frame is not None:
            self.current_frame = frame.copy()

        if detected_posture == "LYING / FALLEN":
            if self.fall_start_time is None:
                self.fall_start_time = current_time

            duration = current_time - self.fall_start_time

            if duration >= self.fall_confirm_seconds:
                if not self.is_alarm_active:
                    self.is_alarm_active = True
                    self.confirmed_fall_time = current_time
                    print("\n[ALERT] *** FALL CONFIRMED! INITIATING COUNTDOWN ***")
                    self.audio.beep(1200, 400)
                    self.audio.speak("Warning, fall detected! Press C to cancel.")

                    # Trigger physical ESP32 Hardware Alarm (Buzzer & Red LED)
                    if self.esp32:
                        self.esp32.set_alert(True)

                # Check emergency dispatch countdown
                if self.confirmed_fall_time:
                    elapsed_emergency = current_time - self.confirmed_fall_time
                    if elapsed_emergency >= self.emergency_dispatch_seconds and not self.emergency_dispatched:
                        self.dispatch_emergency_alert()

                self.state = "FALL_CONFIRMED"
            else:
                self.state = "FALL_WARNING"
        else:
            # Person stood back up safely
            if self.is_alarm_active:
                print(f"[INFO] Person recovered to {detected_posture}. Alarm auto-resetting.")
            self.reset_alarm()
            self.state = detected_posture

    def cancel_alarm_manually(self):
        """Simulates physical cancel button press (or triggered by hardware button)."""
        print("[USER ACTION] False alarm cancel button pressed!")
        self.reset_alarm()
        self.audio.beep(800, 150)
        self.audio.speak("Alarm cancelled.")

    def reset_alarm(self):
        self.fall_start_time = None
        self.confirmed_fall_time = None
        self.is_alarm_active = False
        self.emergency_dispatched = False

        # Reset ESP32 Hardware (Silence Buzzer, Green LED ON)
        if self.esp32:
            self.esp32.set_alert(False)

    def dispatch_emergency_alert(self):
        self.emergency_dispatched = True
        print("\n" + "=" * 60)
        print("[EMERGENCY DISPATCH] >>> DISPATCHING ACTIVE ALERTS (SMS/PUSH/TELEGRAM) <<<")
        print("=" * 60 + "\n")
        self.audio.speak("Emergency contacts have been notified.")
        self.notifier.dispatch_alert(frame=self.current_frame, details=self.last_pose_details)


def draw_hud(frame, detector, posture, angle, aspect_ratio, bbox, keypoints, esp32_bridge=None, cam_info="Webcam"):
    h, w, _ = frame.shape

    # Determine colors based on state
    if detector.state == "FALL_CONFIRMED":
        box_color = (0, 0, 255)       # Red
        status_text = "FALL DETECTED!"
    elif detector.state == "FALL_WARNING":
        box_color = (0, 165, 255)     # Orange
        status_text = "FALL WARNING (VERIFYING...)"
    elif posture == "SITTING":
        box_color = (255, 191, 0)     # Sky Blue
        status_text = "SITTING"
    elif posture == "STANDING":
        box_color = (0, 255, 0)       # Green
        status_text = "STANDING"
    else:
        box_color = (200, 200, 200)   # Gray
        status_text = posture

    # Draw Skeleton Joints and Bones
    if keypoints is not None:
        skeleton_edges = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        for p1_idx, p2_idx in skeleton_edges:
            kp1 = keypoints[p1_idx]
            kp2 = keypoints[p2_idx]
            if kp1[2] > 0.35 and kp2[2] > 0.35:
                cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])),
                         (255, 200, 50), 2)

        for kp in keypoints:
            if kp[2] > 0.35:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1)

    # Draw Bounding Box
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{status_text}", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Top Status Dashboard Banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Text Metrics
    cv2.putText(frame, f"STATE: {detector.state}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    cv2.putText(frame, f"Torso Angle: {angle:.1f} deg", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"Aspect Ratio: {aspect_ratio:.2f}", (240, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Feed & Controls instruction
    cv2.putText(frame, f"Feed: {cam_info}", (w - 290, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
    cv2.putText(frame, "[F] Flip  |  [C] Cancel  |  [Q] Quit", (w - 330, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Bottom Hardware Status Bar (if ESP32 connected)
    if esp32_bridge and esp32_bridge.connected:
        st = esp32_bridge.last_status
        mpu = st.get("mpu", {})
        pir_txt = "MOTION" if st.get("pir", 0) == 1 else "STILL"
        hw_info = f"ESP32 Connected | PIR: {pir_txt} | MPU: {mpu.get('total_g', 1.0):.2f}g (Z={mpu.get('az', 1.0):.2f})"
        cv2.putText(frame, hw_info, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)

    # Emergency Countdown Warning Banner
    if detector.is_alarm_active:
        remaining = 0
        if detector.confirmed_fall_time:
            elapsed = time.time() - detector.confirmed_fall_time
            remaining = max(0.0, detector.emergency_dispatch_seconds - elapsed)

        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 85), (w, h - 25), (0, 0, 200), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)

        if detector.emergency_dispatched:
            alert_msg = "*** EMERGENCY ALERT DISPATCHED TO CONTACTS! ***"
        else:
            alert_msg = f"*** FALL DETECTED! DISPATCH IN {remaining:.1f}s (PRESS 'C' TO CANCEL) ***"

        cv2.putText(frame, alert_msg, (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    elif detector.fall_start_time is not None:
        elapsed = time.time() - detector.fall_start_time
        pct = min(1.0, elapsed / detector.fall_confirm_seconds)
        bar_w = int((w - 40) * pct)
        cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20), (0, 165, 255), -1)
        cv2.putText(frame, f"Verifying posture... {elapsed:.1f}s", (25, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def main():
    print("=" * 65)
    print("  AI IoT Fall Detection System (Webcam & ESP32-CAM)")
    print("=" * 65)

    # Load configuration
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    camera_cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                camera_cfg = json.load(f).get("camera", {})
        except Exception as e:
            print(f"[WARN] Failed to read camera config: {e}")

    cam_type = camera_cfg.get("type", "esp32")
    esp32_ip = camera_cfg.get("esp32_ip", "")
    stream_url = camera_cfg.get("stream_url", "")
    webcam_idx = camera_cfg.get("webcam_index", 0)

    cap = None
    cam_display_label = "Laptop Webcam"

    # Attempt to open ESP32-CAM if configured
    if cam_type == "esp32" and stream_url:
        print(f"[INFO] Connecting to ESP32-CAM stream at: {stream_url}...")
        cap = ThreadedCamera(stream_url, is_stream=True)
        if cap.isOpened():
            cam_host = stream_url.replace("http://", "").split("/")[0]
            cam_display_label = f"ESP32-CAM ({cam_host})"
            print(f"[INFO] Successfully connected to ESP32-CAM ({cam_host}) [Zero-Lag Mode]!")
        else:
            print(f"[WARN] Could not connect to ESP32-CAM at {stream_url}.")
            print("[INFO] Falling back to laptop webcam...")
            cap = None

    if cap is None:
        cap = ThreadedCamera(webcam_idx, is_stream=False)
        cam_display_label = f"Webcam ({webcam_idx})"

    if not cap.isOpened():
        print("[ERROR] Could not open any camera source (ESP32-CAM or Webcam).")
        print("Please check your Wi-Fi stream URL in config.json or verify your webcam.")
        return

    # Detector instance
    detector = FallDetector(
        model_name="yolov8n-pose.pt",
        fall_confirm_seconds=2.5,
        emergency_dispatch_seconds=8.0,
        torso_angle_threshold=42.0,
        aspect_ratio_threshold=0.95
    )

    # ESP32 hardware bridge for Buzzer, LEDs, MPU6050, Button
    esp32_bridge = None
    if esp32_ip:
        print(f"[INFO] Initializing ESP32 hardware bridge for: {esp32_ip}...")
        esp32_bridge = ESP32Bridge(
            ip=esp32_ip,
            on_button_press=lambda: detector.cancel_alarm_manually()
        )
        detector.esp32 = esp32_bridge

    print("\n[READY] Fall detection loop started (Zero-Lag Mode).")
    print("  - Stand, sit, or simulate a fall in view of the camera.")
    print("  - Press 'C' or press the physical ESP32 Button to cancel an alarm.")
    print("  - Press 'Q' to quit.\n")

    win_title = "AI Fall Detection System"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 960, 720)

    flip_180 = camera_cfg.get("flip_180", True)
    prev_frame_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Flip 180 degrees if camera is mounted upside down
        if flip_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Resize if stream is very large
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, 480))

        # Run YOLOv8-pose inference (imgsz=320 optimizes speed to 30+ FPS)
        results = detector.model(frame, conf=0.45, imgsz=320, verbose=False)

        best_posture = "NO PERSON"
        best_angle = 90.0
        best_ratio = 0.5
        best_bbox = None
        best_keypoints = None

        if (len(results) > 0 and
            results[0].boxes is not None and len(results[0].boxes) > 0 and
            results[0].keypoints is not None and len(results[0].keypoints.data) > 0):

            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints_data = results[0].keypoints.data.cpu().numpy()

            if len(boxes) > 0 and len(keypoints_data) > 0:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                primary_idx = int(np.argmax(areas))

                if primary_idx < len(boxes) and primary_idx < len(keypoints_data):
                    best_bbox = boxes[primary_idx]
                    best_keypoints = keypoints_data[primary_idx]

                    posture, angle, ratio, rapid_drop = detector.analyze_pose(best_keypoints, best_bbox)
                    best_posture = posture
                    best_angle = angle
                    best_ratio = ratio

                    if rapid_drop and detector.state != "FALL_CONFIRMED":
                        detector.audio.beep(600, 100)

                    detector.update(posture, angle, ratio, frame=frame)
        else:
            if not detector.is_alarm_active:
                detector.state = "STANDBY (NO PERSON)"

        # Draw HUD
        draw_hud(
            frame, detector, best_posture, best_angle, best_ratio,
            best_bbox, best_keypoints, esp32_bridge=esp32_bridge, cam_info=cam_display_label
        )

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / max(0.001, (curr_time - prev_frame_time))
        prev_frame_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 110, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(win_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('c'), ord('C')):
            detector.cancel_alarm_manually()
        elif key in (ord('f'), ord('F')):
            flip_180 = not flip_180
            print(f"[INFO] Camera orientation toggled: 180-Flip is now {'ON' if flip_180 else 'OFF'}")

        frame_count += 1
        if frame_count > 10:
            try:
                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed cleanly.")


if __name__ == "__main__":
    main()
