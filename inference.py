import os
import pickle

import cv2
import joblib
import mediapipe as mp
import numpy as np

# Load trained model and scaler
model_file = "./model/hand_gesture_model.pkl"
scaler_file = "./model/scaler.pkl"
landmarks_dir = "./landmarks"
class_labels_file = os.path.join(landmarks_dir, "class_labels.pkl")

print("Loading model and configuration...\n")

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    print("Error: Model files not found")
    print("Please run train_model.py first to train the model.")
    exit()

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

with open(class_labels_file, "rb") as f:
    class_labels = pickle.load(f)

print("Model loaded successfully")
print(f"Classes: {class_labels}\n")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Starting real-time hand gesture detection...")
print("Press 'q' to quit\n")

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read frame from webcam")
        break

    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Extract landmark coordinates
            landmarks = hand_landmarks.landmark

            # Get bounding box coordinates
            x_min = min([lm.x for lm in landmarks]) * w
            y_min = min([lm.y for lm in landmarks]) * h
            x_max = max([lm.x for lm in landmarks]) * w
            y_max = max([lm.y for lm in landmarks]) * h

            # Add padding to bounding box
            padding = 20
            x_min = max(0, int(x_min - padding))
            y_min = max(0, int(y_min - padding))
            x_max = min(w, int(x_max + padding))
            y_max = min(h, int(y_max + padding))

            # Create feature vector from landmarks (same as in training)
            landmark_list = []
            for lm in landmarks:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Normalize features
            landmark_array = np.array(landmark_list).reshape(1, -1)
            landmark_scaled = scaler.transform(landmark_array)

            # Predict gesture class
            prediction = model.predict(landmark_scaled)[0]
            confidence = model.predict_proba(landmark_scaled)[0]
            max_confidence = confidence[prediction]

            predicted_class = class_labels[prediction]

            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                          color, thickness)

            # Draw label with background
            label_text = f"{predicted_class}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2

            # Get text size for background
            text_size = cv2.getTextSize(label_text,
                                        font,
                                        font_scale,
                                        font_thickness)[0]
            label_bg_x_min = x_min
            label_bg_y_min = y_min - text_size[1] - 10
            label_bg_x_max = x_min + text_size[0] + 10
            label_bg_y_max = y_min

            # Draw background rectangle for label
            cv2.rectangle(
                frame,
                (label_bg_x_min, label_bg_y_min),
                (label_bg_x_max, label_bg_y_max),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (x_min + 5, y_min - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    else:
        # Show message when no hand is detected
        cv2.putText(
            frame,
            "No hand detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Display frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Gesture recognition closed")
