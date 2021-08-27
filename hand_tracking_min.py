import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# Initialize hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # For drawing the points

while True:
    _, img = cap.read()

    # Convert from BGR (cv2 reads) to RGB (Hands() uses)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # There is at least one hand on the frame
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                height, width, ch = img.shape
                pos_x, pos_y = int(lm.x * width), int(lm.y * height)
                print(id, pos_x, pos_y)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Frames", img)
    cv2.waitKey(1)
