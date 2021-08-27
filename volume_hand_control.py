import cv2
import numpy as np
import hand_tracking_module as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Landmark indexes: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png

cam_width, cam_height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = htm.handDetector(detection_conf = 0.7, max_hands = 1)

# Library for interacting with the os volume controller
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
vol_bar = 400

area = 0
while True:
    _, img = cap.read()

    # Find hand
    img = detector.findHands(img)
    lm_list, bounding_box = detector.findPosition(img, draw = True)

    if len(lm_list) != 0:

        # Filter based on size
        width_box = bounding_box[2] - bounding_box[0]
        height_box = bounding_box[3] - bounding_box[1]
        area = width_box * height_box // 100
        # print(area)

        if 200 < area < 1000:

            # Find distance between index and thumb
            length, img, line_info = detector.findDistance(point1 = 4, point2 = 8, img = img)

            # Convert volume: Hand range: 50 - 200 | Volume range: -65 - 0
            # Convert the hand range to volume range
            vol_bar = np.interp(length, [50, 200], [400, 150])
            vol_per = np.interp(length, [50, 200], [0, 100])

            # Reduce resolution to make it smoother
            smoothness = 10 # For 10 units of volume
            vol_per = smoothness * round(vol_per / smoothness)

            # Check fingers up
            fingers = detector.fingersUp()

            # If pinky finger is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(vol_per / 100, None)
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)



        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cv2.imshow("Frames", img)
    cv2.waitKey(1)


