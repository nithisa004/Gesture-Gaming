import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Safety settings
pyautogui.FAILSAFE = True  
pyautogui.PAUSE = 0.15  

# Mediapipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Gesture tracking variables
prev_index_x = None
prev_index_y = None
movement_threshold = 0.05  

def detect_gesture(landmarks):
    global prev_index_x, prev_index_y

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check for open palm (fingers spread)
    index_to_middle = np.linalg.norm([index_tip.x - middle_tip.x, index_tip.y - middle_tip.y])
    fingers_spread = index_to_middle > 0.1
    open_palm = (
        fingers_spread and
        np.linalg.norm([middle_tip.x - ring_tip.x, middle_tip.y - ring_tip.y]) > 0.1 and
        np.linalg.norm([ring_tip.x - pinky_tip.x, ring_tip.y - pinky_tip.y]) > 0.1
    )

    if open_palm:
        return "Open Palm"

    # Movement-based gestures
    if prev_index_x is None or prev_index_y is None:
        prev_index_x, prev_index_y = index_tip.x, index_tip.y
        return "Neutral"

    delta_x = index_tip.x - prev_index_x
    delta_y = index_tip.y - prev_index_y
    prev_index_x, prev_index_y = index_tip.x, index_tip.y

    if abs(delta_x) > movement_threshold or abs(delta_y) > movement_threshold:
        if abs(delta_x) > abs(delta_y):  
            if delta_x < -movement_threshold:
                return "Left Tilt"
            elif delta_x > movement_threshold:
                return "Right Tilt"
        else:
            if delta_y < -movement_threshold:
                return "Open Hand"
            elif delta_y > movement_threshold:
                return "Slide"

    return "Neutral"

print("Open your game in the browser and focus on the game window. Starting in 5 seconds...")
time.sleep(5)

last_gesture = "Neutral"
frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = "Neutral"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks.landmark)

                if gesture != last_gesture:
                    if gesture == "Open Hand":
                        pyautogui.press('up')        # Jump
                    elif gesture == "Slide":
                        pyautogui.press('down')      # Slide
                    elif gesture == "Left Tilt":
                        pyautogui.press('left')      # Move Left
                    elif gesture == "Right Tilt":
                        pyautogui.press('right')     # Move Right
                    elif gesture == "Open Palm":
                        pyautogui.press('space')     # Use Ability

                    last_gesture = gesture

        # Display current gesture on screen
        cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Control - Press 'q' to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exited by user.")
            break

        frame_count += 1

except KeyboardInterrupt:
    print("Stopped manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()

