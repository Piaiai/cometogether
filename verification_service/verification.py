import os
import sys
import csv
import copy
import logging
import itertools

import cv2
import mediapipe as mp

from hand_gesture_verification.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from verification_service.face_identity import FaceIdentity
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class Verificator:
    def __init__(self, original_image, image_for_verification, original_image_for_torch, verification_image_torch, gesture):
        self.original_image = original_image
        self.original_image_for_torch = original_image_for_torch
        self.image_for_verification = image_for_verification
        self.verification_image_torch = verification_image_torch
        self.gesture = gesture
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.keypoint_classifier = KeyPointClassifier(model_path='hand_gesture_verification/model/keypoint_classifier/keypoint_classifier.tflite')
        self.face_identity = FaceIdentity()
        self.keypoint_classifier_labels = self.read_gesture_classes()

        # point_history_classifier = PointHistoryClassifier()
        self.gesture_idx_to_class_name = {
            0: 'Open',
            1: 'Close',
            2: 'Pointer',
            3: 'OK'
        }
        self.verification_status = {}

    def verify_gesture(self):
        hand_sign_id = None
        image = self.image_for_verification.copy()
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id >= 2:
                    self.verification_status['recozniged_gesture'] = self.gesture_idx_to_class_name[hand_sign_id]
                else:
                    self.verification_status['recozniged_gesture'] = 'Undefined'

                self.verification_status['expected_gesture'] = self.gesture
                self.verification_status['is_correct_gesture'] = self.gesture == self.gesture_idx_to_class_name[hand_sign_id]

        return hand_sign_id

    def verify_face_image(self, threshold=0.58):
        distance = self.face_identity(self.original_image_for_torch, self.verification_image_torch)
        self.verification_status['threshold'] = threshold
        self.verification_status['distance'] = distance
        self.verification_status['face_verified'] = distance <= threshold

        return distance

    def verify_persona(self):
        hand_sign_id = self.verify_gesture()
        distance = self.verify_face_image()
        self.verification_status['person_verified'] = self.verification_status['is_correct_gesture'] and self.verification_status['face_verified']
        return hand_sign_id, distance

    def read_gesture_classes(self):
        with open('hand_gesture_verification/model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        return keypoint_classifier_labels

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
