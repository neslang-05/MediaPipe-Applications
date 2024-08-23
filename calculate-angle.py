import cv2
import mediapipe as mp
import json
import math
import time

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    # Convert the landmark coordinates to vectors
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    
    # Calculate the cosine of the angle between the vectors
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)
    
    # Return the angle in degrees
    angle = math.degrees(math.acos(cos_angle))
    return angle

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
            angles = {}  # Dictionary to store finger angles

            # Calculate angles for each finger using the landmarks
            # Thumb angles
            thumb_angles = calculate_angle(
                [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], 
                [hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z], 
                [hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y, hand_landmarks.landmark[2].z]
            )
            angles['thumb_angle'] = thumb_angles

            # Index finger angles
            index_angles = calculate_angle(
                [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], 
                [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z], 
                [hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z]
            )
            angles['index_angle'] = index_angles

            # Middle finger angles
            middle_angles = calculate_angle(
                [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], 
                [hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z], 
                [hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z]
            )
            angles['middle_angle'] = middle_angles

            # Ring finger angles
            ring_angles = calculate_angle(
                [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], 
                [hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y, hand_landmarks.landmark[13].z], 
                [hand_landmarks.landmark[14].x, hand_landmarks.landmark[14].y, hand_landmarks.landmark[14].z]
            )
            angles['ring_angle'] = ring_angles

            # Pinky finger angles
            pinky_angles = calculate_angle(
                [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], 
                [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z], 
                [hand_landmarks.landmark[18].x, hand_landmarks.landmark[18].y, hand_landmarks.landmark[18].z]
            )
            angles['pinky_angle'] = pinky_angles

            # Add angles and timestamp to hand data
            hand_data["angles"] = angles
            hand_data["timestamp"] = time.time()
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
