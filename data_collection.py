"""
This code collects images for a hand gesture recognition dataset using OpenCV
and MediaPipe.It captures images from the webcam and saves them in class -
specific folders based on user input.The user can start capturing images for
each class by pressing 's' and can stop early by pressing 'q'.The code also
provides feedback on the number of images captured for each class.
"""

import os

import cv2

DATA_DIR = "./data"

class_labels = ["Start", "Stop", "Move_Left", "Move_Right", "Move_Up",
                "Move_Down"]

Number_images = 500

# Create a folder for each class label in the data directory
for labels in class_labels:
    os.makedirs(os.path.join(DATA_DIR, labels))

# Get the list of class folders in the data directory
classfolders = os.listdir(DATA_DIR)

cap = cv2.VideoCapture(0)  # Open the default camera (0)

for classfolder in classfolders:
    image_count = 0

    while image_count < Number_images:
        success, frame = cap.read()
        if not success:
            break

            # Show waiting screen
            cv2.putText(
                frame,
                f"press 's' to start capturing {classfolder}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Images captured: {image_count}/{Number_images}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'Escape' to skip this class",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Frames", frame)

            k = cv2.waitKey(25)
            if k == ord("s"):
                print(f"Started capturing for {classfolder}")
            elif k == 27:
                print(f"Skipped {classfolder}")
                break
        else:
            # Capturing images
            cv2.putText(
                frame,
                f"Capturing {classfolder}: {image_count}/{Number_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'q' to stop capturing early",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Frames", frame)

            # Save image
            imgSavePath = os.path.join(
                DATA_DIR, classfolder, f"{classfolder}_{image_count}.jpg"
            )
            cv2.imwrite(imgSavePath, frame)

            image_count += 1
            print(f"Saved {image_count}/{Number_images} for {classfolder}")

            k = cv2.waitKey(25)
            if k == ord("q"):
                print(f"Stopped early for {classfolder}")
                break

    print(f"Completed {classfolder}: {image_count} images captured\n")

cap.release()
cv2.destroyAllWindows()
