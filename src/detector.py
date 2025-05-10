import mediapipe as mp
import cv2
from constants import MOVEMENT_THRESHOLD


class Detector:
    def __init__(self):
        self.__prev_landmarks = None
        self.__hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def detect_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.__hands.process(frame_rgb)

        # No hand detected
        if not results.multi_hand_landmarks:
            return None, None

        curr_landmarks = results.multi_hand_landmarks[0]

        # Discard blurred frames, e.g. transitions between signs
        movement = self.calc_average_movement(self.__prev_landmarks, curr_landmarks)
        self.__prev_landmarks = curr_landmarks
        if movement > MOVEMENT_THRESHOLD:
            return None, None

        # Get bounding box from landmarks
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in curr_landmarks.landmark]
        y_coords = [lm.y for lm in curr_landmarks.landmark]

        # Convert normalized coords into pixel values
        xmin = int(min(x_coords) * w) - 50
        xmax = int(max(x_coords) * w) + 50
        ymin = int(min(y_coords) * h) - 50
        ymax = int(max(y_coords) * h) + 50

        # Clip to frame bounds
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, w), min(ymax, h)

        # Crop the hand region
        hand_img = frame[ymin:ymax, xmin:xmax]

        return hand_img, ((xmin, ymin), (xmax, ymax))

    @staticmethod
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
