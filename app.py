import cv2
import numpy as np
import collections
import asyncio
from frame_collector import FrameCollector
from classifier import Classifier
from detector import Detector
from constants import INPUT_SIZE, CONFIDENCE_THRESHOLD


async def main():
    classifier = Classifier()
    detector = Detector()
    frame_collector = FrameCollector()
    camera = cv2.VideoCapture(0)
    letters_buffer = collections.deque(maxlen=10)
    last_model_output = ""

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        hand_img = await asyncio.to_thread(detector.detect_image, frame)
        mean_frame = frame_collector.collect_and_return(hand_img)
        if mean_frame is not None:
            cv2.imshow('Model input', mean_frame)
            predicted_label, confidence = await asyncio.to_thread(classifier.classify_image, mean_frame)
            if confidence > CONFIDENCE_THRESHOLD and (
                    len(letters_buffer) == 0 or letters_buffer[-1] != predicted_label):
                letters_buffer.append(predicted_label)
                model_output = ",".join(letters_buffer)
                last_model_output = model_output

        cv2.putText(frame, last_model_output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Live Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


asyncio.run(main())
