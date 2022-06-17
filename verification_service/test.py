import cv2
from verification_service.verification import Verificator

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

image_for_verification = cv2.imread('img.jpg')
original_image = cv2.imread('img2.jpg')

original_image_for_torch = load_image('img2.jpg')
verification_image_torch = load_image('img.jpg')
gesture = 'OK'

verificator = Verificator(original_image, image_for_verification, original_image_for_torch, verification_image_torch, gesture)
verificator.verify_persona()
print(verificator.verification_status)

# keypoint_classifier = KeyPointClassifier()
# hands = mp.solutions.hands.Hands(
#             static_image_mode=True,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5,
# )
# def main():
#     image_for_verification = cv2.imread('img.jpg')
#     image = image_for_verification.copy()
#     image = cv2.flip(image, 1)  # Mirror display
#     debug_image = copy.deepcopy(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = hands.process(image)
#     for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#         landmark_list = calc_landmark_list(debug_image, hand_landmarks)
#         pre_processed_landmark_list = pre_process_landmark(landmark_list)
#         hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#         print(hand_sign_id)
#
#
# def calc_landmark_list(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#
#     landmark_point = []
#
#     # Keypoint
#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         # landmark_z = landmark.z
#
#         landmark_point.append([landmark_x, landmark_y])
#
#     return landmark_point
#
# def pre_process_landmark(landmark_list):
#     temp_landmark_list = copy.deepcopy(landmark_list)
#
#     # Convert to relative coordinates
#     base_x, base_y = 0, 0
#     for index, landmark_point in enumerate(temp_landmark_list):
#         if index == 0:
#             base_x, base_y = landmark_point[0], landmark_point[1]
#
#         temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#         temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
#
#     # Convert to a one-dimensional list
#     temp_landmark_list = list(
#         itertools.chain.from_iterable(temp_landmark_list))
#
#     # Normalization
#     max_value = max(list(map(abs, temp_landmark_list)))
#
#     def normalize_(n):
#         return n / max_value
#
#     temp_landmark_list = list(map(normalize_, temp_landmark_list))
#
#     return temp_landmark_list
#
#
# main()