import cv2
import mediapipe as mp
import math

class handDetector():
    def __init__(self, mode = False, max_hands = 2, detection_conf = 0.5, track_conf = 0.5, tipIds = [4, 8, 12, 16, 20]):
        self.mode = mode
        self.max_hands = 2
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        # Initialize hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils  # For drawing the points

        # Initialize finger tip ids
        self.tipIds = tipIds

    def findHands(self, img, draw = True):
        # Convert from BGR (cv2 reads) to RGB (Hands() uses)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # There is at least one hand on the frame
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self, img, hand_idx = 0, draw = True):
        x_list = []
        y_list = []
        bounding_box = []

        self.lm_list = []
        # There is at least one hand on the frame
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_idx]
            for id, lm in enumerate(my_hand.landmark):
                height, width, ch = img.shape
                pos_x, pos_y = int(lm.x * width), int(lm.y * height)
                self.lm_list.append([id, pos_x, pos_y])

                x_list.append(pos_x)
                y_list.append(pos_y)
                if draw:
                    cv2.circle(img, (pos_x, pos_y), 5, (255, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bounding_box = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

        return self.lm_list, bounding_box

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lm_list[self.tipIds[0]][1] > self.lm_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lm_list[self.tipIds[id]][2] < self.lm_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def findDistance(self, point1, point2, img, draw = True):
        x1, y1 = self.lm_list[point1][1], self.lm_list[point1][2]
        x2, y2 = self.lm_list[point2][1], self.lm_list[point2][2]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, center_x, center_y]


def main():
    cap = cv2.VideoCapture(0)

    detector = handDetector()
    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findPosition(img, draw = False)

        if len(lm_list) != 0:
            print(lm_list[0]) # https://google.github.io/mediapipe/images/mobile/hand_landmarks.png


        cv2.imshow("Frames", img)
        cv2.waitKey(1)


# If we run this script
if __name__ == '__main__':
    main()