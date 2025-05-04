import numpy as np
import cv2
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from constants import INPUT_SIZE, CLASS_INDICES


class Classifier:
    def __init__(self):
        self.index_to_class = {v: k for k, v in CLASS_INDICES.items()}
        self.__model = load_model("./model-training/model-train-augment-finetuned.keras")
        # Warmup model
        self.__model.predict(np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32), verbose=0)

    def classify_image(self, image):
        input_image = cv2.resize(image, INPUT_SIZE)
        input_image = preprocess_input(input_image)
        input_image = np.expand_dims(input_image, axis=0)

        prediction = self.__model.predict(input_image, verbose=0)
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)
        predicted_label = self.index_to_class[predicted_index]
        return predicted_label, confidence
