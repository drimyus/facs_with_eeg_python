import cv2
import numpy as np
import dlib
import os
import sys
import math


# dlib's landmarks
NOSE_POINTS = [4]
RIGHT_EYE_POINTS = list(range(2, 4))
LEFT_EYE_POINTS = list(range(0, 2))
NUM_TOTAL_POINTS = 5


def convert_rect(cv_rect):
    (x, y, w, h) = cv_rect.astype(dtype=np.long)
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    return rect


class Face:
    def __init__(self, detect_mode='haar'):

        # location of detector model
        cur = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(cur, os.pardir))
        detector_dir = os.path.join(root, "model/detector")

        self.detect_mode = detect_mode
        # init the dlib's face detector
        if detect_mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
        elif detect_mode == 'haar':
            # init the opencv's cascade haarcascade
            front_detector_path = os.path.join(detector_dir, "haarcascade_frontalface_alt2.xml")
            profile_detector_path = os.path.join(detector_dir, "haarcascade_profileface.xml")
            if not os.path.isfile(front_detector_path) or not os.path.isfile(profile_detector_path):
                sys.stderr.write("no exist detector.\n")
                sys.exit(1)

            self.front_detector = cv2.CascadeClassifier(front_detector_path)
            self.profile_detector = cv2.CascadeClassifier(profile_detector_path)

        else:
            sys.stderr.write("no defined detector mode.\n")
            sys.exit(1)

        # init the dlib's face shape predictor
        detector_path = os.path.join(detector_dir, "shape_predictor_68_face_landmarks.dat")
        if not os.path.isfile(detector_path):
            sys.stderr.write("no exist shape_predictor.\n")
            sys.exit(1)
        self.shape_predictor = dlib.shape_predictor(detector_path)

        recognizer_path = os.path.join(detector_dir, "dlib_face_recognition_resnet_model_v1.dat")
        if not os.path.isfile(recognizer_path):
            sys.stderr.write("no exist shape predictor.\n")
            sys.exit(1)
        self.recognizer = dlib.face_recognition_model_v1(recognizer_path)

    def detect_face(self, image):
        if self.detect_mode == 'dlib':
            rects = self.detector(image, 0)
            return rects
        elif self.detect_mode == 'haar':
            front_rects = self.front_detector.detectMultiScale(image, scaleFactor=1.5, minNeighbors=5)
            profile_rects = self.profile_detector.detectMultiScale(image, scaleFactor=1.5, minNeighbors=5)

            cv_rects = list(front_rects) + list(profile_rects)
            rects = []
            # convert cv_rect to dlib_rect
            for cv_rect in cv_rects:
                rect = convert_rect(cv_rect)
                rects.append(rect)
            return rects

    def recog_description(self, face):

        h, w = face.shape[:2]
        face_rect = dlib.rectangle(int(0), int(0), int(w), int(h))

        shape = self.shape_predictor(face, face_rect)

        face_description = self.recognizer.compute_face_descriptor(face, shape)

        return np.asarray(face_description)

