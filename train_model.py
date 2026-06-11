import os
import pickle

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E501
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load landmarks and labels
landmarks_dir = "./landmarks"
landmarks_file = os.path.join(landmarks_dir, "landmarks.npy")
labels_file = os.path.join(landmarks_dir, "labels.npy")
class_labels_file = os.path.join(landmarks_dir, "class_labels.pkl")

print("Loading extracted landmarks...\n")

X = np.load(landmarks_file)
y = np.load(labels_file)

with open(class_labels_file, "rb") as f:
    class_labels = pickle.load(f)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Classes: {class_labels}\n")

# Split data into training and testing sets
print("Splitting data into training (80%) and testing (20%)...\n")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features (important for better model performance)
print("Normalizing features...\n")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
print("This may take a moment depending on dataset size...\n")

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

clf.fit(X_train_scaled, y_train)

# Make predictions
print("Evaluating model...\n")
y_pred = clf.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"{'='*50}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}\n")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save the trained model and scaler
model_dir = "./model"
os.makedirs(model_dir, exist_ok=True)

model_file = os.path.join(model_dir, "hand_gesture_model.pkl")
scaler_file = os.path.join(model_dir, "scaler.pkl")

joblib.dump(clf, model_file)
joblib.dump(scaler, scaler_file)

print(f"\n{'='*50}")
print(f"✓ Model saved to {model_file}")
print(f"✓ Scaler saved to {scaler_file}")
print(f"{'='*50}")
