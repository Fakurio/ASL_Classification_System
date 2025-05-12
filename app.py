import cv2
import collections
import asyncio
import time
from classifier import Classifier
from detector import Detector
from constants import CONFIDENCE_THRESHOLD, INTERVAL_THRESHOLD, CLASS_INDICES


def get_map_predictions_to_letters():
    prediction_to_letter = {}
    for letter in CLASS_INDICES.keys():
        match letter:
            case "space":
                prediction_to_letter["space"] = "_"
            case "nothing":
                prediction_to_letter["nothing"] = ""
            case _:
                prediction_to_letter[letter] = letter
    return prediction_to_letter


async def main():
    classifier = Classifier()
    detector = Detector()
    camera = cv2.VideoCapture(0)
    letters_buffer = collections.deque(maxlen=10)
    model_output = ""
    last_correct_prediction_time = time.time()
    prediction_to_letter = get_map_predictions_to_letters()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        current_time = time.time()
        hand_img, border_coords = await asyncio.to_thread(detector.detect_image, frame)

        if hand_img is not None:
            cv2.rectangle(frame, border_coords[0], border_coords[1], (255, 0, 0), 2)
            predicted_label, confidence = await asyncio.to_thread(classifier.classify_image, hand_img)
            if confidence > CONFIDENCE_THRESHOLD and current_time - last_correct_prediction_time > INTERVAL_THRESHOLD:
                last_correct_prediction_time = time.time()
                predicted_letter = prediction_to_letter[predicted_label]
                if predicted_letter == "del":
                    letters_buffer.clear()
                else:
                    letters_buffer.append(predicted_letter)
                model_output = "".join(letters_buffer)

        cv2.putText(frame, model_output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press C to remove last letter", (10, 440), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Live Camera Feed', frame)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('c') and len(letters_buffer) > 0:
            letters_buffer.pop()
            model_output = "".join(letters_buffer)
        if key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


asyncio.run(main())
