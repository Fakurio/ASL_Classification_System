import cv2
import numpy as np
from constants import FRAMES_COUNT, INPUT_SIZE


class FrameCollector:
    def __init__(self):
        self.__frames = []

    def collect_and_return(self, hand_img):
        if hand_img is None:
            self.__frames.clear()
            return None

        if len(self.__frames) < FRAMES_COUNT:
            hand_img = cv2.resize(hand_img, INPUT_SIZE)
            self.__frames.append(hand_img.astype(np.float32))
            return None

        # List full -> return mean frame and add frame from current call to empty list
        mean_frame = np.mean(self.__frames, axis=0).astype(np.uint8)
        self.__frames.clear()
        hand_img = cv2.resize(hand_img, INPUT_SIZE)
        self.__frames.append(hand_img.astype(np.float32))
        return mean_frame
