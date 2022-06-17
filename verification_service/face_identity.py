from keras import backend as K
from keras.layers import Input, Layer
import numpy as np
import os.path
from face_verification.model_face import create_model
from verification_service.align import AlignDlib


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


class FaceIdentity:
    def __init__(self):
        # self.nn4_small2 = create_model()
        # self.metadata = load_metadata('images')
        # self.in_a = Input(shape=(96, 96, 3))
        # self.in_p = Input(shape=(96, 96, 3))
        # self.in_n = Input(shape=(96, 96, 3))
        #
        # self.emb_a = self.nn4_small2(self.in_a)
        # self.emb_p = self.nn4_small2(self.in_p)
        # self.emb_n = self.nn4_small2(self.in_n)
        # self.triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([self.emb_a, self.emb_p, self.emb_n])

        self.nn4_small2_pretrained = create_model()
        self.nn4_small2_pretrained.load_weights('face_verification/weights/nn4.small2.v1.h5')
        self.alignment = AlignDlib('face_verification/weights/models/landmarks.dat')


    def __call__(self, original_image, verification_image, *args, **kwargs):
        original_image_aligned = self.align_image(original_image)
        original_image_aligned = (original_image_aligned / 255).astype(np.float32)
        original_image_embedding = self.nn4_small2_pretrained.predict(np.expand_dims(original_image_aligned, axis=0))[0]

        verification_image_aligned = self.align_image(verification_image)
        verification_image_aligned = (verification_image_aligned / 255).astype(np.float32)
        verification_image_embedding = self.nn4_small2_pretrained.predict(np.expand_dims(verification_image_aligned, axis=0))[0]

        return self.distance(original_image_embedding, verification_image_embedding)

    def align_image(self, img):
        return self.alignment.align(96, img, self.alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def distance(self, emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

    def train(self):
        pass