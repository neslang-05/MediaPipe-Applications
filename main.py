import cv2
import mediapipe as mp
import json

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start Video Capture
cap = cv2.VideoCapture(1)

gesture_data = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = {}
            for id, lm in enumerate(hand_landmarks.landmark):
                hand_data[f"landmark_{id}"] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                }
            gesture_data.append(hand_data)

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the gesture data to a JSON file
with open('gesture_data.json', 'w') as json_file:
    json.dump(gesture_data, json_file, indent=4)

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
