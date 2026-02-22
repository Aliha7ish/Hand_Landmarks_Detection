import cv2
import numpy as np
import joblib
import mediapipe as mp
import argparse
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="camera",
                    help="camera | video | image")
parser.add_argument("--path", type=str,
                    help="Path to video or image file")
parser.add_argument("--display_width", type=int, default=800,
                    help="Width to resize frame for display")
args = parser.parse_args()


# Load trained pipeline
models_dir = "./models"
rf_pipeline = joblib.load(f"{models_dir}/random_forest_pipeline.pkl")
le = joblib.load(f"{models_dir}/transformers/label_encoder.pkl")


# Create Hand Landmarker
model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(options)


# Buffers for smoothing
prediction_buffer = deque(maxlen=10)
landmark_buffer = deque(maxlen=5)


# Input Source Selection
if args.mode == "camera":
    cap = cv2.VideoCapture(0)
elif args.mode == "video":
    if args.path is None:
        raise ValueError("Provide --path for video mode")
    cap = cv2.VideoCapture(args.path)
elif args.mode == "image":
    if args.path is None:
        raise ValueError("Provide --path for image mode")
    frame = cv2.imread(args.path)
    cap = None
else:
    raise ValueError("Mode must be camera, video, or image")


# Frame Processing Function
def process_frame(frame, display_width=args.display_width):
    global prediction_buffer, landmark_buffer

    # Resize frame to fixed display width (keep aspect ratio)
    h, w, _ = frame.shape
    scale_ratio = display_width / w
    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)
    frame = cv2.resize(frame, (new_w, new_h))
    h, w, _ = frame.shape

    # Prepare MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            points = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), max(2, w//200), (0, 255, 0), -1)

            # Draw connections
            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (255, 0, 0), max(1, w//400))

            # Convert landmarks
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks])

            # Landmark smoothing
            if args.mode != "image":
                landmark_buffer.append(landmarks)
                smoothed_landmarks = np.mean(landmark_buffer, axis=0)
            else:
                smoothed_landmarks = landmarks

            # Prediction
            pred_label = rf_pipeline.predict(smoothed_landmarks)[0]
            pred_text = le.inverse_transform([pred_label])[0]

            if args.mode != "image":
                prediction_buffer.append(pred_text)
                smoothed_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
            else:
                smoothed_prediction = pred_text

            # Display prediction text with background for visibility
            font_scale = max(0.8, w/600)      # clamp minimum font size
            thickness = max(2, w//300)       # clamp thickness
            text = f"Prediction: {smoothed_prediction}"

            # get text size
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            margin = 10
            x, y = margin, text_h + margin

            # draw background rectangle
            cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0,0,0), -1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    return frame


if args.mode == "image":
    processed = process_frame(frame)
    cv2.imshow("Hand Gesture Recognition", processed)
    cv2.waitKey(0)
else:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip camera feed horizontally for "mirror" view
        if args.mode == "camera":
            frame = cv2.flip(frame, 1)

        processed = process_frame(frame)
        cv2.imshow("Hand Gesture Recognition", processed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

cv2.destroyAllWindows()