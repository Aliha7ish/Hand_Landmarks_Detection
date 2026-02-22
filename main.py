import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from statistics import mode


# Load trained pipeline
models_dir = "./models"
rf_pipeline = joblib.load(f"{models_dir}/random_forest_pipeline.pkl")
le = joblib.load(f"{models_dir}/transformers/label_encoder.pkl")

# Hand Landmarker object
model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)


# apply mood smoothing
# we will take the last 10 frames to take the most frequent prediction
prediction_buffer = deque(maxlen=10)

# we will take the last 5 frames to take the most frequent landmaks 
# to solve the problem of landmarks jitter
landmark_buffer = deque(maxlen=5)


# Live camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape

            points = []

            # Draw landmark points
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections (manually)
            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]

            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

            # Convert landmarks for prediction
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks]).flatten()
            # landmarks = landmarks.reshape(1, -1)

            # append landmarks to buffer
            landmark_buffer.append(landmarks)

            if len(landmark_buffer) > 0:
                smoothed_landmarks = np.mean(landmark_buffer, axis=0)
            else:
                smoothed_landmarks = landmarks
            
            # reshape smoothed_landmarks to be row vector
            smoothed_landmarks = smoothed_landmarks.reshape(1, -1)


            # normalize the landmarks then predict the numerical label 
            pred_label = rf_pipeline.predict(smoothed_landmarks)[0]
            # convert the numerical label into the corsponding categorical one
            pred_text = le.inverse_transform([pred_label])[0]

            # add predicted gesture to buffer
            prediction_buffer.append(pred_text)

            # apply mode smoothing (select the most frequent gesture even in tie)
            if len(prediction_buffer) > 0:
                smoothed_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
            else:
                smoothed_prediction = pred_text

            cv2.putText(
                frame,
                f"Prediction: {smoothed_prediction}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()