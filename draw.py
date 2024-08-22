import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Canvas

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the Tkinter window
root = tk.Tk()
root.title("Gesture-Based Drawing")
canvas = Canvas(root, width=640, height=480, bg='white')
canvas.pack()

shapes = []

def draw_shape(shape, x, y):
    if shape == 'circle':
        shapes.append(canvas.create_oval(x-25, y-25, x+25, y+25, outline='black', width=2))
    elif shape == 'square':
        shapes.append(canvas.create_rectangle(x-25, y-25, x+25, y+25, outline='black', width=2))
    elif shape == 'rectangle':
        shapes.append(canvas.create_rectangle(x-50, y-25, x+50, y+25, outline='black', width=2))
    elif shape == 'triangle':
        shapes.append(canvas.create_polygon(x, y-30, x-30, y+30, x+30, y+30, outline='black', width=2, fill=''))

def erase_shapes():
    for shape in shapes:
        canvas.delete(shape)
    shapes.clear()

# Start Video Capture
cap = cv2.VideoCapture(1)

def update():
    success, frame = cap.read()
    if not success:
        return

    # Flip the frame horizontally for a later selfie-view display
    # frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Detect index finger tip (landmark id 8)
                if id == 8:
                    draw_shape('circle', cx, cy)

                # Detect palm center (approx. landmark id 0)
                if id == 0:
                    erase_shapes()

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Update the Tkinter window with OpenCV frame
    cv2.imshow("Gesture Tracking", frame)
    root.after(10, update)

# Start the Tkinter loop
root.after(10, update)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
