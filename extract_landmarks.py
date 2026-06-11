import os
import pickle

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)

DATA_DIR = "./data"
OUTPUT_DIR = "./landmarks"
class_labels = ["Start", "Stop", "Move_Left", "Move_Right", "Move_Up",
                "Move_Down"]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

landmarks_data = []
labels_data = []

print("Extracting hand landmarks from images...\n")

# Iterate through each class folder
for class_idx, class_label in enumerate(class_labels):
    class_dir = os.path.join(DATA_DIR, class_label)

    if not os.path.exists(class_dir):
        print(f" Skipping {class_label} - folder not found")
        continue

    images = os.listdir(class_dir)
    processed = 0
    skipped = 0

    print(f"Processing {class_label}...")

    for img_name in images:
        if not img_name.endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_dir, img_name)

        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                skipped += 1
                continue

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:  # noqa: E501
                # Extract landmarks from the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]

                # Flatten landmark coordinates into a single array
                # Each landmark has x, y, z coordinates
                landmark_list = []
                for landmark in hand_landmarks.landmark:
                    landmark_list.extend([landmark.x, landmark.y, landmark.z])

                landmarks_data.append(landmark_list)
                labels_data.append(class_idx)
                processed += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"  Error processing {img_name}: {e}")
            skipped += 1

    print(f"  ✓ Processed: {processed}, Skipped: {skipped}\n")

# Convert to numpy arrays
X = np.array(landmarks_data)
y = np.array(labels_data)

print(f"\n{'='*50}")
print(f"Total images with hand landmarks: {len(landmarks_data)}")
print(f"Feature shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"{'='*50}\n")

# Save landmarks and labels
landmarks_file = os.path.join(OUTPUT_DIR, "landmarks.npy")
labels_file = os.path.join(OUTPUT_DIR, "labels.npy")
class_labels_file = os.path.join(OUTPUT_DIR, "class_labels.pkl")

np.save(landmarks_file, X)
np.save(labels_file, y)

with open(class_labels_file, "wb") as f:
    pickle.dump(class_labels, f)

print(f"✓ Landmarks saved to {landmarks_file}")
print(f"✓ Labels saved to {labels_file}")
print(f"✓ Class labels saved to {class_labels_file}")

hands.close()
