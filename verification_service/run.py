from verification_service import come_together_pb2
from verification_service.come_together_pb2_grpc import *
from PIL import Image

from concurrent import futures
import logging
import sys
import io
import cv2
from PIL import ImageFile
from verification import Verificator
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    stream=sys.stdout,
    level=logging.DEBUG
)


class VerificationServiceServicerImplementation(VerificationServiceServicer):
    def __init__(self):
        self.original_image_save_name = 'original_image.jpg'
        self.validation_image_save_name = 'validation_image.jpg'
        self.map_idx_to_gesture_name = {1: 'OK', 2: 'Pointer'}
        super().__init__()

    def validate_image(self, gesture='OK'):
        logging.info('Initializing verification')
        image_for_verification = cv2.imread(self.validation_image_save_name)
        original_image = cv2.imread(self.original_image_save_name)

        original_image_for_torch = self.load_image(self.original_image_save_name)
        verification_image_torch = self.load_image(self.validation_image_save_name)

        verificator = Verificator(original_image, image_for_verification, original_image_for_torch,
                                  verification_image_torch, gesture)
        verificator.verify_persona()

        print(f'Verification complete. Results: {verificator.verification_status}')
        return verificator.verification_status

    def load_image(self, path):
        img = cv2.imread(path, 1)
        # OpenCV loads images with color channels
        # in BGR order. So we need to reverse them
        return img[..., ::-1]

    def ValidatePhoto(self, request_iterator, context):
        logging.info(f'Validating photo')
        original_photo_bytes = bytearray()
        validation_photo_bytes = bytearray()
        gesture = None
        for i, data in enumerate(request_iterator):
            gesture = data.gestue
            if not data.validation_photo_complete:
                validation_photo_bytes.extend(data.validation_photo)
            if not data.user_photo_complete:
                original_photo_bytes.extend(data.user_photo)

            if data.user_photo_complete and data.validation_photo_complete:
                logging.info('Read images, saving')
                break

        original_image = Image.open(io.BytesIO(original_photo_bytes))
        validation_photo = Image.open(io.BytesIO(validation_photo_bytes))
        original_image.save(self.original_image_save_name)
        validation_photo.save(self.validation_image_save_name)

        result = self.validate_image(self.map_idx_to_gesture_name[gesture])

        res = 0
        if result['person_verified']:
            res = 1
        elif not result['is_correct_gesture']:
            res = 2
        elif not result['face_verified']:
            res = 3

        return come_together_pb2.validate_photo_response(res=res)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_VerificationServiceServicer_to_server(
        VerificationServiceServicerImplementation(), server)
    server.add_insecure_port('0.0.0.0:8080')
    logging.info("Server is starting")
    server.start()
    server.wait_for_termination()

serve()