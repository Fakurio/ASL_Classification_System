import cv2
import time
import os
import mediapipe as mp
import numpy as np
import collections
import asyncio
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

INPUT_SIZE = (224, 224)
CLASS_INDICES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                 'M': 12,
                 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
                 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}

index_to_class = {v: k for k, v in CLASS_INDICES.items()}
model = load_model("./model-training/model-train-augment-finetuned.keras")
# Warm up model
model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))


def classify_image(image, ):
    input_image = cv2.resize(image, INPUT_SIZE)
    input_image = preprocess_input(input_image)
    input_image = np.expand_dims(input_image, axis=0)

    prediction = model.predict(input_image, verbose=0)
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_label = index_to_class[predicted_index]
    return predicted_label, confidence


def calc_average_movement(prev_landmarks, curr_landmarks):
    if prev_landmarks is None or curr_landmarks is None:
        return float('inf')

    movement = 0
    for p1, p2 in zip(prev_landmarks.landmark, curr_landmarks.landmark):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        movement += (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    return movement / len(curr_landmarks.landmark)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)


def detect_image(frame, prev_landmarks):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            curr_landmarks = results.multi_hand_landmarks[0]
            # Discard blurred frames, eg. transitions between signs
            movement = calc_average_movement(prev_landmarks, curr_landmarks)
            if movement < 0.01:
                # Get bounding box from landmarks
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                xmin = int(min(x_coords) * w) - 100
                xmax = int(max(x_coords) * w) + 100
                ymin = int(min(y_coords) * h) - 100
                ymax = int(max(y_coords) * h) + 100

                # Clip to frame bounds
                xmin, ymin = max(xmin, 0), max(ymin, 0)
                xmax, ymax = min(xmax, w), min(ymax, h)

                # Crop the hand region
                hand_img = frame[ymin:ymax, xmin:xmax]

                return hand_img, curr_landmarks

            return None, curr_landmarks

    return None, None


async def main():
    FRAMES_COUNT = 5  # Number of frames to average
    frames = []
    prev_landmarks = None
    letters_buffer = collections.deque(maxlen=10)
    last_model_output = ""

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        hand_img, landmarks = await asyncio.to_thread(detect_image, frame, prev_landmarks)
        if hand_img is not None:
            if len(frames) < FRAMES_COUNT:
                hand_img = cv2.resize(hand_img, INPUT_SIZE)
                frames.append(hand_img.astype(np.float32))
            else:
                mean_frame = np.mean(frames, axis=0).astype(np.uint8)
                predicted_label, confidence = await asyncio.to_thread(classify_image, mean_frame)
                if confidence > 0.75 and (len(letters_buffer) == 0 or letters_buffer[-1] != predicted_label):
                    letters_buffer.append(predicted_label)
                    output_text = ",".join(letters_buffer)
                    last_model_output = output_text
                cv2.imshow('Model input', mean_frame)
                frames = []
        else:
            frames = []

        if landmarks:
            prev_landmarks = landmarks

        cv2.putText(frame, last_model_output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Live Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


asyncio.run(main())
