#!/usr/bin/env python3
"""
Raspberry Pi Code - Fire Detection System with BlueDot Motor Control
Controls motors via BlueDot | Pump relay now handled by Arduino via Serial
NOW WITH AI/ML Fire Detection using trained Random Forest model
+ Pump Speed Control via Laptop Gesture Controller (UDP port 9999)
FIX: Pump turns OFF automatically when fire stops
FIX v2: Corrected motor direction logic for L298N wiring
NOTE: Relay is connected to Arduino — Pi no longer controls it directly.

WIRING (L298N):
  IN1 -> GPIO23 | IN2 -> GPIO24 | ENA -> GPIO18  (Right side motors: OUT1+, OUT2-)
  IN3 -> GPIO27 | IN4 -> GPIO22 | ENB -> GPIO13  (Left  side motors: OUT3+, OUT4-)

  RIGHT wheels: +ve -> OUT1, -ve -> OUT2
  LEFT  wheels: +ve -> OUT3, -ve -> OUT4

FORWARD  : Right (IN1=H, IN2=L) | Left (IN3=H, IN4=L)
BACKWARD : Right (IN1=L, IN2=H) | Left (IN3=L, IN4=H)
TURN LEFT : Right forward (IN1=H, IN2=L) | Left  backward (IN3=L, IN4=H)
TURN RIGHT: Right backward(IN1=L, IN2=H) | Left  forward  (IN3=H, IN4=L)

NOTE: If the car still goes in the wrong direction after flashing this code,
      physically swap the OUT1/OUT2 wires OR OUT3/OUT4 wires on the L298N
      (a mirrored motor mount is the most common hardware cause).
"""

import serial
import time
import RPi.GPIO as GPIO
from bluedot import BlueDot
from signal import pause
import threading
import joblib
import numpy as np
import socket

# ── Detection mode + image fire status globals ───
detection_mode    = 1             # 1=sensors, 2=image, 3=both (set from PUMP.py)
image_fire_status = "IMAGE_SAFE"  # updated by image_alert_receiver_thread

# -------- Serial Configuration --------
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE   = 9600

# ──────────────────────────────────────────────────────────────
# MOTOR PIN SETUP  (matches your L298N connections exactly)
# ──────────────────────────────────────────────────────────────
ENA = 18   # PWM speed – RIGHT side (OUT1 / OUT2)
IN1 = 23   # RIGHT motor direction pin A
IN2 = 24   # RIGHT motor direction pin B

ENB = 13   # PWM speed – LEFT side (OUT3 / OUT4)
IN3 = 27   # LEFT motor direction pin A
IN4 = 22   # LEFT motor direction pin B

# -------- GPIO Setup --------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in [ENA, IN1, IN2, ENB, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)   # safe default: all LOW

# -------- PWM setup for motors --------
pwmA = GPIO.PWM(ENA, 1000)   # Right side
pwmB = GPIO.PWM(ENB, 1000)   # Left  side
pwmA.start(0)
pwmB.start(0)

# -------- Pump PWM Speed Pin --------
PUMP_PWM_PIN = 25
GPIO.setup(PUMP_PWM_PIN, GPIO.OUT)
pump_speed_pwm = GPIO.PWM(PUMP_PWM_PIN, 1000)
pump_speed_pwm.start(0)

# ──────────────────────────────────────────────────────────────
# SERVO SETUP  (GPIO12 — hardware PWM)
# Servo expects 50 Hz signal:
#   0°   = ~2.5% duty cycle
#   90°  = ~7.5% duty cycle
#   180° = ~12.5% duty cycle
# ──────────────────────────────────────────────────────────────
SERVO_PIN = 12
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(SERVO_PIN, 50)   # 50 Hz for servo
servo_pwm.start(7.5)                  # start at centre (90°)

def angle_to_duty(angle):
    """Convert 0-180° angle to duty cycle for standard servo."""
    return 2.5 + (angle / 180.0) * 10.0

def servo_set_angle(angle):
    """Move servo to given angle (0-180)."""
    angle = max(0, min(180, angle))
    servo_pwm.ChangeDutyCycle(angle_to_duty(angle))

# -------- Global State --------
gesture_speed     = 0
fire_active       = False
was_fire_active   = False
current_pwm_speed = 0

# -------- Servo State --------
servo_locked_angle = 90      # angle servo is currently locked at
servo_sweeping     = False  # <-- ADD THIS # True while auto-sweep is running
current_servo_angle = 90   
servo_state_lock   = threading.Lock()
latest_flame_ao    = 1023    # shared from sensor thread; lower = more fire

# ======================================================
# SERVO CONTROL
# ======================================================
def servo_set_angle(angle):
    """Move servo to given angle (0-180)."""
    angle = max(0, min(180, int(angle)))
    duty  = 2.5 + (angle / 180.0) * 10.0
    servo_pwm.ChangeDutyCycle(duty)

def servo_sweep_and_lock():
    """
    Sweeps servo 0-180, records flame_ao at each step,
    then locks at the angle with the LOWEST analog value (strongest fire).
    Called in a background thread when fire is first detected.
    """
    global servo_sweeping, servo_locked_angle, latest_flame_ao
    with servo_state_lock:
        if servo_sweeping:
            return
        servo_sweeping = True
    print('[SERVO] Sweeping 0-180 to find fire direction...')
    best_angle = 90
    best_value = 9999
    for angle in range(0, 181, 10):
        servo_set_angle(angle)
        time.sleep(0.15)
        reading = latest_flame_ao
        print(f'[SERVO] {angle:>3} deg -> flame_ao = {reading}')
        if reading < best_value:
            best_value = reading
            best_angle = angle
    print(f'[SERVO] Locking at {best_angle} deg (flame_ao={best_value})')
    servo_set_angle(best_angle)
    servo_locked_angle = best_angle
    with servo_state_lock:
        servo_sweeping = False

def servo_return_center():
    """Return servo to 90 deg center when fire clears."""
    global servo_locked_angle
    print('[SERVO] Fire cleared - returning to center (90 deg)')
    servo_set_angle(90)
    servo_locked_angle = 90


    """
    Continuously sweeps servo 0<->180 while rover is moving normally.
    Automatically pauses when fire is active or a fire-sweep is running.
    """
def servo_auto_sweep_thread():
    global current_servo_angle

    direction = 1
    angle     = 90
    current_servo_angle = angle

    while True:
        if fire_active:
            time.sleep(0.05)
            continue

        angle += direction * 2

        if angle >= 180:
            angle = 180
            direction = -1
        elif angle <= 0:
            angle = 0
            direction = 1

        servo_set_angle(angle)
        current_servo_angle = angle   # <-- TRACK CURRENT ANGLE

        time.sleep(0.03)

def set_pump(speed):
    """Set pump PWM speed (0-100). Relay is handled by Arduino."""
    global current_pwm_speed
    speed = max(0, min(100, int(speed)))
    if speed == 0:
        pump_speed_pwm.ChangeDutyCycle(0)
        current_pwm_speed = 0
        print("Pump OFF")
    else:
        if current_pwm_speed == 0:
            time.sleep(0.05)
        pump_speed_pwm.ChangeDutyCycle(speed)
        current_pwm_speed = speed
        print(f"Pump speed set to {speed}%")

# ======================================================
# ML MODEL LOADING
# ======================================================
MODEL_PATH   = '/home/pi/fire_project/fire_model.pkl'
SCALER_PATH  = '/home/pi/fire_project/scaler.pkl'
ENCODER_PATH = '/home/pi/fire_project/label_encoder.pkl'

print("Loading AI Fire Detection Model...")
try:
    fire_model    = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✓ AI Model loaded successfully!")
    ML_AVAILABLE = True
except Exception as e:
    print(f"✗ Warning: Could not load AI model: {e}")
    print("  Will use basic threshold-based detection instead.")
    ML_AVAILABLE = False

# ======================================================
# ML FIRE DETECTION
# ======================================================
def predict_fire_status_ml(mq2, mq7, mq135, flame, temperature, humidity):
    try:
        features        = np.array([[mq2, mq7, mq135, flame, temperature, humidity]])
        features_scaled = scaler.transform(features)
        prediction      = fire_model.predict(features_scaled)
        raw_label = label_encoder.inverse_transform(prediction)[0]

        # Normalise whatever label the encoder returns into FIRE/WARNING/SAFE
        raw_upper = str(raw_label).upper()
        if "FIRE" in raw_upper:
            return "FIRE"
        elif "WARN" in raw_upper or "SMOKE" in raw_upper or "DANGER" in raw_upper:
            return "WARNING"
        else:
            return "SAFE"
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None

def basic_fire_detection(data):
    if data['flame'] == 0:          # digital flame sensor triggered
        return "FIRE"
    elif data['flame_ao'] < 500:    # analog flame value low = strong flame signal
        return "FIRE"
    elif data['temperature'] > 55 or data['mq2'] > 500:
        return "WARNING"
    return "SAFE"

def check_fire_status(data):
    if ML_AVAILABLE:
        result = predict_fire_status_ml(
            data['mq2'], data['mq7'], data['mq135'],
            data['flame'], data['temperature'], data['humidity']
        )
        if result is not None:
            return result
    return basic_fire_detection(data)

# ======================================================
# COMBINE SENSOR + IMAGE STATUS
# ======================================================
def get_combined_status(sensor_status):
    """Mode 1=sensors only, 2=image only, 3=both"""
    global detection_mode, image_fire_status
    if detection_mode == 1:
        return sensor_status
    elif detection_mode == 2:
        if   image_fire_status == "IMAGE_FIRE":    return "FIRE"
        elif image_fire_status == "IMAGE_WARNING": return "WARNING"
        else:                                      return "SAFE"
    elif detection_mode == 3:
        if   image_fire_status == "IMAGE_FIRE"    or sensor_status == "FIRE":    return "FIRE"
        elif image_fire_status == "IMAGE_WARNING" or sensor_status == "WARNING": return "WARNING"
        else:                                                                     return "SAFE"
    return sensor_status

# ======================================================
# FIRE RESPONSE
# ======================================================
def handle_fire_status(status):
    global fire_active, was_fire_active

    if status == "FIRE":
        print("\n" + "!"*60)
        print("!!! ALERT: AI DETECTED FIRE !!!")
        print("!!! STOPPING ROVER + ACTIVATING WATER PUMP !!!")
        print("!"*60 + "\n")
        fire_active     = True
        was_fire_active = True
        stop()
        # [SERVO] Trigger sweep in background so pump activates immediately
        servo_set_angle(current_servo_angle)  # <-- LOCK AT CURRENT POSITION
        set_pump(100)

    elif status == "WARNING":
        print("\n" + "~"*60)
        print("~~~ WARNING: Smoke/Heat Detected - Be Alert ~~~")
        print("~"*60 + "\n")
        fire_active = False
        if was_fire_active:
            print("Fire cleared — turning pump OFF automatically.")
            set_pump(0)
            servo_return_center()
            was_fire_active = False

    elif status == "SAFE":
        fire_active = False
        if was_fire_active:
            print("Fire cleared — turning pump OFF automatically.")
            set_pump(0)
            servo_return_center()
            was_fire_active = False

# ======================================================
# PUMP SPEED RECEIVER THREAD
# ======================================================
def pump_speed_receiver_thread():
    global gesture_speed

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9999))
    sock.settimeout(1.0)

    last_applied_speed = -1
    print("✓ Pump speed receiver listening on port 9999...")

    while True:
        try:
            data, addr = sock.recvfrom(16)
            speed = int(data.decode().strip())
            speed = max(0, min(100, speed))
            gesture_speed = speed

            if not fire_active:
                if speed != last_applied_speed:
                    set_pump(speed)
                    last_applied_speed = speed
                    print(f"Gesture speed applied: {speed}%")
            else:
                print(f"Fire active — ignoring gesture ({speed}%)")

        except socket.timeout:
            pass
        except Exception as e:
            print(f"Speed receiver error: {e}")

# ── Image alert receiver ─────────────────────────────
def image_alert_receiver_thread():
    global image_fire_status
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 9998))
    sock.settimeout(1.0)
    print("✓ Image alert receiver on port 9998...")
    while True:
        try:
            data, _ = sock.recvfrom(32)
            status = data.decode().strip()
            if status in ["IMAGE_FIRE", "IMAGE_WARNING", "IMAGE_SAFE"]:
                image_fire_status = status
                if status != "IMAGE_SAFE":
                    print(f"[CAMERA] {status}")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Image receiver error: {e}")

# ── Mode receiver ─────────────────────────────────────
def mode_receiver_thread():
    global detection_mode
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9997))
    sock.settimeout(1.0)
    print("✓ Mode receiver on port 9997...")
    while True:
        try:
            data, _ = sock.recvfrom(8)
            mode = int(data.decode().strip())
            if mode in [1, 2, 3]:
                detection_mode = mode
                names = {1: "Sensors Only", 2: "Image Only", 3: "Both"}
                print(f"Detection mode → {mode} ({names[mode]})")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Mode receiver error: {e}")

# -------- Serial Functions --------
def setup_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("✓ Serial connection established with Arduino")
        return ser
    except serial.SerialException as e:
        print(f"✗ Error: Could not open serial port {SERIAL_PORT}: {e}")
        return None

def read_sensor_data(ser):
    try:
        # Read directly — blocks until a line arrives (up to serial timeout=1s)
        # Do NOT gate on in_waiting; that caused stale/frozen readings.
        line = ser.readline().decode('utf-8').strip()
        if not line:
            return None  # skip empty lines silently
        data = line.split(',')
        if len(data) != 7:
            return None  # skip non-data lines (status messages etc.)
        # Ensure none of the fields are empty strings
        if any(d.strip() == '' for d in data):
            return None
        # Arduino sends: mq2, mq7, mq135, flame_do, flame_ao, temperature, humidity
        return {
            'mq2':         int(data[0]),
            'mq7':         int(data[1]),
            'mq135':       int(data[2]),
            'flame':       int(data[3]),   # digital: 0=fire, 1=no fire
            'flame_ao':    int(data[4]),   # analog flame value
            'temperature': float(data[5]),
            'humidity':    float(data[6])
        }
    except ValueError:
        pass  # silently ignore malformed/status lines
    except Exception as e:
        print(f"Error reading sensor data: {e}")
    return None

def display_sensor_data(data, status):
    print("\n" + "="*55)
    print("SENSOR READINGS  |  AI STATUS: " + status)
    print("="*55)
    print(f"MQ-2  (Smoke/LPG)    : {data['mq2']}")
    print(f"MQ-7  (CO)           : {data['mq7']}")
    print(f"MQ-135(Air Quality)  : {data['mq135']}")
    print(f"Flame Sensor (Digital): {'🔥 FIRE DETECTED!' if data['flame'] == 0 else 'No Fire'}")
    print(f"Flame Sensor (Analog) : {data['flame_ao']}")
    print(f"Temperature          : {data['temperature']} °C")
    print(f"Humidity             : {data['humidity']} %")
    print(f"AI Prediction        : >>> {status} <<<")
    print(f"Pump Speed (Gesture) : {gesture_speed}%")
    print(f"Pump PWM Speed       : {current_pwm_speed}%")
    print(f"Servo Angle          : {servo_locked_angle} deg {'[LOCKED ON FIRE]' if fire_active else '[SWEEPING]'}")
    print("="*55)

# ──────────────────────────────────────────────────────────────
# MOTOR CONTROL  — Fixed direction logic
#
#   RIGHT side: IN1=H, IN2=L → forward  |  IN1=L, IN2=H → backward
#   LEFT  side: IN3=H, IN4=L → forward  |  IN3=L, IN4=H → backward
#
# If car goes BACKWARD when you press FORWARD, swap the
# OUT1/OUT2 wires (or OUT3/OUT4 wires) on the L298N board.
# ──────────────────────────────────────────────────────────────
def forward(speed=70):
    """Both sides drive forward."""
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)   # Right fwd
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)   # Left  fwd
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def backward(speed=70):
    """Both sides drive backward."""
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)  # Right bwd
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)  # Left  bwd
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def turn_left(speed=70):
    """Right side forward, left side backward → car turns LEFT."""
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)   # Right fwd
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)  # Left  bwd
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def turn_right(speed=70):
    """Left side forward, right side backward → car turns RIGHT."""
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)  # Right bwd
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)   # Left  fwd
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def stop():
    """Cut all motor signals and PWM."""
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

# ──────────────────────────────────────────────────────────────
# BLUEDOT HANDLERS
# BlueDot coordinate system:
#   y > 0  → finger in top half    → FORWARD
#   y < 0  → finger in bottom half → BACKWARD
#   x > 0  → finger on right       → RIGHT
#   x < 0  → finger on left        → LEFT
# ──────────────────────────────────────────────────────────────
def move(pos):
    global fire_active
    if fire_active:
        return

    x     = pos.x
    y     = pos.y
    speed = max(30, min(int(pos.distance * 100), 100))

    if y > 0.3:                       # Forward zone
        if abs(x) < 0.3:
            forward(speed)            # Straight ahead
        elif x > 0:
            turn_right(speed)         # Forward + lean right
        else:
            turn_left(speed)          # Forward + lean left
    elif y < -0.3:                    # Backward zone
        backward(speed)
    elif abs(x) > 0.3:               # Side-only zone (pure turn)
        if x > 0:
            turn_right(speed)
        else:
            turn_left(speed)
    else:
        stop()

def pressed():
    print("BlueDot pressed!")

def released():
    global fire_active
    if not fire_active:
        print("BlueDot released - stopping rover")
        stop()

# -------- Sensor Thread --------
def sensor_monitoring_thread(ser):
    global fire_active, latest_flame_ao
    last_data   = None
    last_status = "SAFE"
    try:
        while True:
            data = read_sensor_data(ser)
            if data:
                last_data      = data
                latest_flame_ao = data['flame_ao']   # share with servo thread
                last_status    = get_combined_status(check_fire_status(data))
                display_sensor_data(last_data, last_status)
                fire_active = (last_status == "FIRE")
                handle_fire_status(last_status)
            elif last_data:
                # No new line yet — re-display last known values so screen isn't frozen
                display_sensor_data(last_data, last_status)
            # No sleep — readline() already blocks up to serial timeout (1s)
    except Exception as e:
        print(f"\nError in sensor monitoring thread: {e}")

# -------- Main --------
def main():
    print("="*60)
    print("  Fire Detection Rover - AI Powered")
    print("  BlueDot Motor Control + ML Fire Detection")
    print("  + Gesture Pump Speed Control from Laptop")
    print("  (Relay now controlled by Arduino)")
    print("="*60)

    print("\n1. Connecting to Arduino via Serial...")
    ser = setup_serial()
    if ser is None:
        print("Failed to connect to Arduino. Exiting.")
        return

    print("\n2. Starting AI sensor monitoring thread...")
    sensor_thread = threading.Thread(
        target=sensor_monitoring_thread, args=(ser,), daemon=True
    )
    sensor_thread.start()
    print("   ✓ Sensor thread running")

    print("\n3. Starting pump speed receiver thread...")
    speed_thread = threading.Thread(target=pump_speed_receiver_thread, daemon=True)
    speed_thread.start()

    print("\n4. Starting image alert receiver thread...")
    img_thread = threading.Thread(target=image_alert_receiver_thread, daemon=True)
    img_thread.start()

    print("\n5. Starting detection mode receiver thread...")
    mode_thread = threading.Thread(target=mode_receiver_thread, daemon=True)
    mode_thread.start()

    print("\n6. Starting servo auto-sweep thread...")
    servo_thread = threading.Thread(target=servo_auto_sweep_thread, daemon=True)
    servo_thread.start()
    print("   servo sweeping 0<->180 continuously")

    print("\n7. Starting BlueDot motor controller...")
    bd               = BlueDot()
    bd.when_pressed  = pressed
    bd.when_moved    = move
    bd.when_released = released

    print("\n" + "="*60)
    print("  SYSTEM READY")
    print(f"  BlueDot Server: {bd.server.server_address}")
    print("  Open BlueDot app on your phone and connect!")
    print("  Run pump_gesture_controller.py on your laptop!")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")

    try:
        pause()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop()
        set_pump(0)
        pump_speed_pwm.stop()
        servo_set_angle(90)
        servo_pwm.stop()
        pwmA.stop()
        pwmB.stop()
        if ser:
            ser.close()
        GPIO.cleanup()
        print("✓ Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
