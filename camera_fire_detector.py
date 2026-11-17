
#!/usr/bin/env python3
"""
camera_fire_detector.py  -  Pi Camera fire checker (runs as a SEPARATE script)

How it works
  * Sits idle with the camera OFF.
  * Rover script sends "CAM_ON"  (UDP 127.0.0.1:9996) when sensors look suspicious.
  * Camera starts, every frame is checked for fire-coloured pixels.
  * Result is sent to the rover on UDP 127.0.0.1:9998 as:
        IMAGE_SAFE / IMAGE_WARNING / IMAGE_FIRE
    (your existing image_alert_receiver_thread already listens for these)
  * Rover sends "CAM_OFF" when sensors are normal again -> camera stops,
    and IMAGE_SAFE is sent so the rover never stays stuck in "fire".

Run normally :  python3 camera_fire_detector.py
Test camera  :  python3 camera_fire_detector.py --test      (camera always on, prints values)
"""

import os
import sys
import time
import socket
import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------- Settings ----------------
CONTROL_ADDR = ("127.0.0.1", 9996)     # receives CAM_ON / CAM_OFF
ALERT_ADDR   = ("127.0.0.1", 9998)     # sends IMAGE_* to rover

FRAME_SIZE   = (320, 240)              # small = fast on Pi

# Fire colour range in HSV (red-orange-yellow, bright and saturated)
HSV_LOWER = np.array([0,   120, 200])
HSV_UPPER = np.array([35,  255, 255])

FIRE_RATIO_WARNING = 0.010   # >=1% of the picture looks like fire  -> WARNING
FIRE_RATIO_FIRE    = 0.040   # >=4% of the picture looks like fire  -> FIRE candidate
CONSECUTIVE_FRAMES = 3       # FIRE must be seen this many frames in a row

# Optional: real flames flicker, static orange objects don't.
# Turn ON only if you get false alarms from orange/red objects.
USE_FLICKER_CHECK  = False
FLICKER_MIN        = 0.02

SAVE_SNAPSHOTS     = True
SNAPSHOT_DIR       = "/home/pi/fire_project/fire_snapshots"
SNAPSHOT_EVERY_SEC = 5

TEST_MODE = "--test" in sys.argv
# ------------------------------------------

alert_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_status(status):
    try:
        alert_sock.sendto(status.encode(), ALERT_ADDR)
    except Exception as e:
        print(f"Could not send status: {e}")

def start_camera():
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}   # RGB888 gives BGR order for OpenCV
    )
    cam.configure(config)
    cam.start()
    time.sleep(1.0)   # let exposure settle
    print("Camera started")
    return cam

def stop_camera(cam):
    try:
        cam.stop()
        cam.close()
    except Exception:
        pass
    print("Camera stopped")

def analyse_frame(frame, prev_mask):
    """Returns (fire_ratio, flicker_ratio, mask)."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    fire_pixels = cv2.countNonZero(mask)
    fire_ratio  = fire_pixels / float(mask.shape[0] * mask.shape[1])

    flicker_ratio = 0.0
    if prev_mask is not None and fire_pixels > 0:
        diff = cv2.absdiff(mask, prev_mask)
        flicker_ratio = cv2.countNonZero(diff) / float(fire_pixels)
    return fire_ratio, flicker_ratio, mask

def save_snapshot(frame, mask):
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        name = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"{name}_frame.jpg"), frame)
        cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"{name}_mask.jpg"),  mask)
    except Exception as e:
        print(f"Snapshot error: {e}")

def main():
    ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctl.bind(CONTROL_ADDR)

    cam            = None
    prev_mask      = None
    fire_streak    = 0
    last_snapshot  = 0.0

    if TEST_MODE:
        print("TEST MODE: camera always on. Ctrl+C to quit.")
        cam = start_camera()
    else:
        print(f"Waiting for CAM_ON on {CONTROL_ADDR} ... (camera is OFF)")
        send_status("IMAGE_SAFE")

    try:
        while True:
            # ---- listen for ON/OFF commands ----
            ctl.settimeout(0.01 if cam else 0.5)
            try:
                data, _ = ctl.recvfrom(16)
                cmd = data.decode().strip()
            except socket.timeout:
                cmd = None

            if cmd == "CAM_ON" and cam is None:
                cam = start_camera()
                prev_mask, fire_streak = None, 0
            elif cmd == "CAM_OFF" and cam is not None and not TEST_MODE:
                stop_camera(cam)
                cam = None
                send_status("IMAGE_SAFE")      # clear any old fire status
                print("Waiting for CAM_ON ... (camera is OFF)")
                continue

            if cam is None:
                continue

            # ---- analyse one frame ----
            frame = cam.capture_array()
            fire_ratio, flicker_ratio, mask = analyse_frame(frame, prev_mask)
            prev_mask = mask

            looks_like_fire = fire_ratio >= FIRE_RATIO_FIRE
            if USE_FLICKER_CHECK and looks_like_fire:
                looks_like_fire = flicker_ratio >= FLICKER_MIN

            fire_streak = fire_streak + 1 if looks_like_fire else 0

            if fire_streak >= CONSECUTIVE_FRAMES:
                status = "IMAGE_FIRE"
            elif fire_ratio >= FIRE_RATIO_WARNING:
                status = "IMAGE_WARNING"
            else:
                status = "IMAGE_SAFE"

            send_status(status)

            if TEST_MODE or status != "IMAGE_SAFE":
                print(f"{status:14s} fire_ratio={fire_ratio*100:5.2f}%  "
                      f"flicker={flicker_ratio:4.2f}  streak={fire_streak}")

            if (SAVE_SNAPSHOTS and status == "IMAGE_FIRE"
                    and time.time() - last_snapshot > SNAPSHOT_EVERY_SEC):
                save_snapshot(frame, mask)
                last_snapshot = time.time()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if cam is not None:
            stop_camera(cam)
        send_status("IMAGE_SAFE")

if __name__ == "__main__":
    main()
