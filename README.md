🛡️ AI-Powered IoT Fall Detection System for Elderly Care
An intelligent, low-cost IoT and computer vision-based fall detection system designed to safeguard elderly individuals living alone. This project combines edge hardware sensors with a Python-powered computer vision backend to detect sudden falls, filter out false alarms, and instantly dispatch emergency alerts with visual proof via Telegram.
🚀 Key Features
 * Real-Time Video Monitoring: Streams live video via ESP32-CAM to monitor rooms continuously.
 * Multi-Sensor Fusion: Integrates an ESP32-CAM, MPU6050 accelerometer/gyroscope, and PIR motion sensor for reliable movement and orientation tracking.
 * Smart Fall Detection Algorithm: Analyzes bounding box dimensions (comparing length vs. breadth ratios) and torso angles to distinguish between normal standing/sitting and actual falls.
 * False Alarm Mitigation (8-Second Grace Period): If a potential fall is flagged, an 8-second countdown timer starts. If the person recovers or presses the physical cancel button, the alarm auto-resets.
 * Instant Emergency Alerts: If unaddressed, it captures a snapshot of the incident and dispatches an emergency notification directly to a designated Telegram Bot complete with timestamps, location details, and image proof.
 * Status Indication & Control: Features Red/Green status LEDs, a buzzer for local alerts, and a physical push button for immediate manual reset.
🛠️ Hardware Components
| Component | Function |
|---|---|
| ESP32-CAM | Captures live video feed and streams it to the Python backend. |
| ESP32 / Microcontroller | Core processing and communication hub. |
| MPU6050 | 6-axis accelerometer & gyroscope for motion/tilt tracking. |
| PIR Motion Sensor | Detects human presence and motion in the room. |
| Push Button | Physical panic/reset button to cancel false alarms locally. |
| Buzzer | Emits an audible alarm during unconfirmed/confirmed falls. |
| Green & Red LEDs | Visual status indicators (Safe/Standby vs. Alert/Fall). |
| Breadboard & Jumper Wires | Prototyping circuit connections. |
| Power Supply | USB powered / 18650 Lithium-ion rechargeable battery setup. |
💻 Software & Tech Stack
 * Language: Python
 * Computer Vision & Processing: OpenCV, JSON data handling
 * Communication & Notifiers: Telegram Bot API (python-telegram-bot / HTTP requests)
 * IDE / Firmware: Arduino IDE (for ESP32 programming)
⚙️ System Workflow & Logic
 * Standby State: The system monitors the room using the PIR sensor and live video feed from the ESP32-CAM (Torso Angle: ~90° indicates standing/sitting).
 * Trigger Condition: When a sudden change in posture occurs (e.g., breadth exceeds length or torso angle drops significantly), the system flags a potential fall.
 * Grace Period & Verification: * A 8-second countdown is initiated.
   * Local buzzer sounds and the Red LED turns on.
   * If the person stands back up or someone presses the physical Reset Button, the system returns to normal standby.
 * Emergency Dispatch: If the timer expires without cancellation, the system captures an image snapshot, logs the event details (time, location, torso angle), and triggers the Telegram Bot to send an emergency alert with the photo attached.
📊 System Visuals & Dashboard
1. Live Monitoring Interface
The desktop GUI tracking torso angles, aspect ratios, PIR motion status, and MPU sensor telemetry in real-time.
2. Terminal Logs & Alert Dispatch
Backend console logs showing fall confirmation, countdown handling, and successful Telegram dispatch.
3. Telegram Emergency Notification
Instant alert received on mobile devices containing the emergency timestamp, room location, sensor metrics, and image proof.
📂 Project Circuit & Wiring Setup
The hardware prototype integrates the ESP32-CAM, sensors, LEDs, buzzer, and push button on breadboards connected via jumper wires. Ensure all common grounds (GND) are tied together, and power inputs are regulated (3.3V / 5V as required by individual modules).
🚀 Getting Started & Installation
1. Hardware Assembly
 * Assemble the circuit following your schematic layout, connecting the ESP32-CAM, MPU6050 via I2C, PIR sensor to digital pins, and the push button / LEDs / buzzer with appropriate current-limiting resistors.
2. Flash the ESP32-CAM Firmware
 * Open the ESP32-CAM code in the Arduino IDE.
 * Update your Wi-Fi credentials (SSID and PASSWORD).
 * Flash the code to the ESP32-CAM module to stream video over your local network.
3. Run the Python Backend
 * Clone the repository and install dependencies:
   pip install opencv-python requests numpy

 * Configure your Telegram Bot Token and Chat ID in the python script.
 * Run the main application:
   python main.py

🛡️ License
This project is open-source and intended for educational and safety assistive technology purposes. Feel free to fork, improve, and adapt for elderly care solutions!
# Fall_Detection_system