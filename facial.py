import cv2
import pandas as pd
from fer import FER
import time

# Initialize the FER facial expression detector
detector = FER()

# Initialize Video Capture
cap = cv2.VideoCapture(1)

# Initialize the list to store expression data
expression_data = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Detect facial expressions in the frame
    result = detector.detect_emotions(frame)

    for face in result:
        (x, y, w, h) = face["box"]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the expression and its confidence
        emotions = face["emotions"]
        emotion, score = max(emotions.items(), key=lambda item: item[1])

        # Display the emotion on the frame
        cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Log the data
        expression_data.append({
            "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Emotion": emotion,
            "Confidence": score,
            "Details": emotions
        })

    # Display the frame with detected expressions
    cv2.imshow("Facial Expression Recognition", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the expression data to a CSV file
df = pd.DataFrame(expression_data)
df.to_csv('expression_data.csv', index=False)

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
