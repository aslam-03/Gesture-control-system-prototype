import cv2
import mediapipe as mp
import pyautogui
import time


def count_fingers(lst):
    cn = 0

    # Calculate threshold based on the palm height
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    # Check finger positions against the threshold
    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cn += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cn += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cn += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cn += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cn += 1

    return cn


# Initialize video capture
cap = cv2.VideoCapture(0)

# MediaPipe hands and drawing utilities
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

prev_count = -1
start_time = None

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:  # Check if the frame was captured properly
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    res = hand_obj.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        hand_keypoints = res.multi_hand_landmarks[0]
        cnt = count_fingers(hand_keypoints)

        if prev_count != cnt:
            if start_time is None:
                start_time = time.time()
            elif (time.time() - start_time) > 0.2:
                if cnt == 1:
                    pyautogui.press("right")
                elif cnt == 2:
                    pyautogui.press("left")
                elif cnt == 3:
                    pyautogui.press("up")
                elif cnt == 4:
                    pyautogui.press("down")
                elif cnt == 5:
                    pyautogui.press("space")

                prev_count = cnt
                start_time = None  # Reset timer after action

        # Draw hand landmarks on the frame
        drawing.draw_landmarks(frame, hand_keypoints, hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
