from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
import tkinter
import tkinter.filedialog
import dlib
import cv2
import sys
import os
import random
from PIL import Image, ExifTags

if sys.version_info[0] == 3:  # python 3x
    from src.face import Face
elif sys.version_info[0] == 2:  # python 2x
    from face import Face


FACE_DATASET = "crop"
TRAIN_DATASET = "train"
TEST_DATASET = "test"


class StandAlone:
    def __init__(self, dataset, model_path, stand_flag=True):

        # location of classifier model
        self.model = None
        self.model_path = model_path

        # location of dataset
        self.dataset = dataset
        self.crop_dataset = os.path.join(dataset, FACE_DATASET)
        self.train_dataset = os.path.join(dataset, TRAIN_DATASET)
        self.test_dataset = os.path.join(dataset, TEST_DATASET)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.ensmeble_ratio = 0.75  # trains : tests = 3 : 1

        self.stand_flag = stand_flag
        self.dlib_face = Face('dlib')
        self.haar_face = Face('haar')

        self.face_width = 151
        self.face_height = 151

        self.rect_color = (0, 255, 0)
        self.text_color = (255, 255, 0)

        # load the model
        sys.stdout.write("Loading the model.\n")
        success = self.load()

        # if not success:
        #     # No exist trained model, so training...
        #     self.train_model()

    def load(self):
        if os.path.isfile(self.model_path):
            try:
                # loading
                self.model = joblib.load(self.model_path)
                return True
            except Exception as ex:
                print(ex)
        else:
            sys.stdout.write("    No exist Model {}, so it should be trained\n".format(self.model_path))

    def load_image(self, file_path):

        try:
            image = Image.open(file_path)

            orientation = None
            for key in ExifTags.TAGS.keys():
                if ExifTags.TAGS[key] == 'Orientation':
                    orientation = key
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
            # image.save(file_path)
            # image.close()

            cv_img = np.array(image)
            cv_img = cv_img[:, :, ::-1].copy()
            return np.array(cv_img)

        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            cv_img = cv2.imread(file_path)
            return cv_img

    def calib_orie(self, image):

        face = self.dlib_face

        max_rects = []
        max_image = image
        for rotate_code in range(3):
            rot_image = cv2.rotate(image, rotateCode=rotate_code)
            rects = face.detect_face(rot_image)
            if len(rects) > len(max_rects):
                max_rects = rects
                max_image = rot_image

        return max_rects, max_image

    def ensemble_data(self):
        crop_dataset = self.crop_dataset

        train_dataset = self.train_dataset
        if not os.path.isdir(train_dataset):
            os.mkdir(train_dataset)
        
        test_dataset = self.test_dataset
        if not os.path.isdir(test_dataset):
            os.mkdir(test_dataset)

        sys.stdout.write("Ensembiling the data.\n")

        if not os.path.isdir(crop_dataset):
            sys.stderr.write("\tNo exist such director: {}\n".format(crop_dataset))
            sys.exit(1)
        sys.stdout.write("\tdataset: {}\n".format(crop_dataset))

        """ counting """
        sys.stdout.write("\tCount the # files(faces) on dataset.\n")
        persons = []
        cnts = []
        for dirpath, dirnames, filenames in os.walk(crop_dataset):
            dirnames.sort()
            for subdirname in dirnames:
                subdirpath = os.path.join(dirpath, subdirname)

                cnts.append(len(os.listdir(subdirpath)))
                persons.append(subdirname)

                sys.stdout.write("\t\tperson: {}, images: {}\n".format(subdirname, len(os.listdir(subdirpath))))
            break

        """ ensembling """
        sys.stdout.write("\tBalace the dataset.\n")
        min_cnt = min(cnts)

        for person in persons:
            subdirpath = os.path.join(crop_dataset, person)
            files = os.listdir(subdirpath)

            samples = random.sample(files, min_cnt)  # pickle the random items from the list
            for sample in samples:
                src = os.path.join(subdirpath, sample)
                if samples.index(sample) <= self.ensmeble_ratio * len(samples):  # for training
                    new_subdirpath = os.path.join(train_dataset, person)
                    if not os.path.isdir(new_subdirpath):
                        os.mkdir(new_subdirpath)
                    dst = os.path.join(new_subdirpath, sample)

                else:  # for testing
                    new_subdirpath = os.path.join(test_dataset, person)
                    if not os.path.isdir(new_subdirpath):
                        os.mkdir(new_subdirpath)
                    dst = os.path.join(new_subdirpath, sample)

                crop = cv2.imread(src)
                if self.stand_flag:
                    stand_img = self.standardize_face(crop)
                else:
                    stand_img = crop

                cv2.imwrite(dst, stand_img)
                cv2.imshow("face", crop)
                cv2.imshow("stand", stand_img)

        sys.stdout.write("\nEnsembled!\n")

    def standardize_face(self, face):

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        clahe_image = self.clahe.apply(gray)

        stand = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

        stand = cv2.resize(stand, (self.face_width, self.face_height))

        return stand

    def image_descriptions(self, image, face):

        if image.shape[:2] == (self.face_height, self.face_width):
            rects = [dlib.rectangle(int(0), int(0), int(image.shape[1]), int(image.shape[0]))]
        else:
            rects = face.detect_face(image)

        descriptions = []
        calib_image = None

        if len(rects) == 0:
            calib_image = self.calib_orie(image)
        else:
            calib_image = image

        for rect in rects:
            crop = calib_image[max(0, rect.top()): max(image.shape[0], rect.bottom()),
                               max(rect.left(), 0):min(rect.right(), image.shape[1])]

            if self.stand_flag:
                stand_face = self.standardize_face(crop)
            else:
                stand_face = crop

            description = self.dlib_face.recog_description(stand_face)
            descriptions.append(description)

        return calib_image, descriptions, rects

    def train_data(self, raw_data):

        dataset = self.crop_dataset
        if not os.path.isdir(dataset):
            os.mkdir(dataset)
        sys.stdout.write("Preparing the face dataset from the raw images.\n")

        if not os.path.isdir(raw_data):
            sys.stderr.write("\tCan not find such directory: {}\n".format(raw_data))
        if not os.path.isdir(dataset):
            sys.stdout.write("\tNo exist such director, so will create new directory: {}\n".format(dataset))
            os.mkdir(dataset)

        sys.stdout.write("\tsource:      {}\n".format(raw_data))
        sys.stdout.write("\tdestination: {}\n".format(dataset))

        sys.stdout.write("\tScaning the source directory.\n")
        for dirpath, dirnames, filenames in os.walk(raw_data):
            dirnames.sort()
            for subdirname in dirnames:
                subdirpath = os.path.join(dirpath, subdirname)
                new_subdirpath = os.path.join(dataset, subdirname)
                if not os.path.isdir(new_subdirpath):
                    os.mkdir(new_subdirpath)

                for filename in os.listdir(subdirpath):
                    sys.stdout.write("\r\t\t{} / {}".format(subdirname, filename))
                    sys.stdout.flush()

                    crop = None
                    image = self.load_image(os.path.join(subdirpath, filename))

                    if image.shape[:2] == (self.face_height, self.face_width):
                        crop = image
                    else:
                        # cropping the face from the image
                        # and resizing
                        rects = self.dlib_face.detect_face(image)
                        # find the correct orientation
                        if len(rects) == 0:
                            rects, image = self.calib_orie(image)

                        if len(rects) != 0:
                            (x, y, w, h) = (rects[0].left(), rects[0].top(), rects[0].right() - rects[0].left(),
                                            rects[0].bottom() - rects[0].top())

                            height, width = image.shape[:2]

                            crop = image[max(0, y): min(y + h, height), max(0, x):min(width, x + w)]
                            crop = cv2.resize(crop, (self.face_width, self.face_height))

                    if crop is not None:
                        cv2.imwrite(filename=os.path.join(new_subdirpath, filename), img=crop)
                        cv2.imshow("face", crop)
                        cv2.waitKey(1)
                    else:
                        sys.stdout.write("\t no face: {} / {}\n".format(subdirpath, filename))
        sys.stdout.write("\nCropped!\n")

    def train_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        else:
            self.model_path = model_path
        
        dataset = self.train_dataset

        sys.stdout.write("Training the model.\n")

        if not os.path.isdir(dataset):
            sys.stderr.write("\tNo exist Dataset for training{}\n".format(dataset))
            exit(1)

        # initialize the data matrix and labels list
        data = []
        labels = []

        """-----------------------------------------------------------------------------------------"""
        sys.stdout.write("\tScanning the dataset.\n")
        # loop over the input images
        for dirpath, dirnames, filenames in os.walk(dataset):
            dirnames.sort()
            for subdirname in dirnames:
                subject_path = os.path.join(dirpath, subdirname)

                for filename in os.listdir(subject_path):
                    sys.stdout.write("\r\t\tscanning: {} {}".format(subject_path, filename))
                    sys.stdout.flush()

                    img = self.load_image(os.path.join(subject_path, filename))
                    _, descriptions, rects = self.image_descriptions(img, self.dlib_face)

                    if len(descriptions) == 0:
                        continue

                    label, hist = subdirname, descriptions[0]  # get label, histogram

                    data.append(hist)
                    labels.append(label)

        """-----------------------------------------------------------------------------------------"""
        sys.stdout.write("\nConfigure the SVM model.\n")
        # Configure the model : linear SVM model with probability capabilities
        model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                    tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                    decision_function_shape='ovr', random_state=None)
        # model = SVC(C=1.0, kernel='linear')
        # model = SVC(probability=True)

        # Train the model
        model.fit(data, labels)

        joblib.dump(model, self.model_path)

        """-----------------------------------------------------------------------------------------"""
        sys.stdout.write("\tFinished the training.\n")
        self.model = model

    def check_precision(self, dir_path):

        confuse_mat = []
        total = 0
        num_pos = 0
        num_neg = 0

        sys.stdout.write("Check the precision with dataset {}\n".format(dir_path))

        if not os.path.isdir(dir_path):
            sys.stderr.write("\tCan not find such directory: {}\n".format(dir_path))
            sys.exit(1)

        sys.stdout.write("\tsource:      {}\n".format(dir_path))

        sys.stdout.write("\tScaning the source directory.\n")
        for dirpath, dirnames, filenames in os.walk(dir_path):
            dirnames.sort()

            for subdirname in dirnames:
                prec_line = np.zeros(len(dirnames), dtype=np.uint8).tolist()

                subdirpath = os.path.join(dirpath, subdirname)
                for filename in os.listdir(subdirpath):
                    sys.stdout.write("\r\t\tscanning: {} {}".format(subdirname, filename))
                    sys.stdout.flush()

                    img = self.load_image(os.path.join(subdirpath, filename))

                    _, descriptions, rects = self.image_descriptions(img, self.dlib_face)

                    if len(descriptions) == 0:
                        continue

                    fid, idx, probs = self.classify_description(descriptions[0])
                    if fid is not None:
                        prec_line[idx] += 1
                        if idx == dirnames.index(subdirname):
                            num_pos += 1
                        else:
                            num_neg += 1
                        total += 1

                prec_line.append(subdirname)
                prec_line.append(len(os.listdir(subdirpath)))

                confuse_mat.append(prec_line)

        sys.stdout.write(
            "\ntotal: {},  positive: {},  negative: {},  precision:{}\n".format(total, num_pos, num_neg,
                                                                                float(num_pos) / float(total)))
        for line in confuse_mat:
            print(line)

    def classify_description(self, description):
        try:
            description = description.reshape(1, -1)

            # Get a prediction from the model including probability:
            probab = self.model.predict_proba(description)

            max_ind = np.argmax(probab)

            # Rearrange by size
            sort_probab = np.sort(probab, axis=None)[::-1]

            if sort_probab[0] / sort_probab[1] < 0.7:
                predlbl = "UnKnown"
            else:
                predlbl = self.model.classes_[max_ind]

            return predlbl, max_ind, probab
        except Exception as e:
            sys.stdout.write(str(e) + "\n")
            pass

    def test_image_file(self):
        root = tkinter.Tk()
        root.withdraw()
        select_file = (tkinter.filedialog.askopenfile(initialdir='.', title='Please select a image file'))
        image_path = select_file.name
        root.update()

        try:
            # Loop to recognize faces
            image = self.load_image(image_path)

            calib_image, descriptions, rects = self.image_descriptions(image, self.haar_face)

            if len(descriptions) == 0:
                sys.stdout.write("No face image\n")
                sys.exit(1)
            else:
                for i in range(len(rects)):
                    description = descriptions[i]
                    rect = rects[i]

                    fid, idx, probs = self.classify_description(description)

                    cv2.rectangle(calib_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), self.rect_color, 3)
                    cv2.putText(calib_image, fid, (rect.left(), rect.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 3)

                show_image = cv2.resize(calib_image, (int(max(calib_image.shape[1] / 4, 130)), int(max(calib_image.shape[0] / 4, 150))))
                cv2.imshow(image_path[-20:], show_image)

                sys.stdout.write("[{}]    id: {}\n".format(fid, str(idx)))
                print(probs)

                cv2.waitKey(0)

        except Exception as e:
            sys.stdout.write(str(e) + "\n")


if __name__ == '__main__':

    model = "../model/classifier/model.pkl"
    root_dataset = "../dataset"
    st = StandAlone(dataset=root_dataset, model_path=model)

    raw_images_folder = "../dataset/rawdata"
    st.train_data(raw_data=raw_images_folder)
    st.ensemble_data()
    st.train_model()

    check_dataset = "../dataset/crop"
    st.check_precision(check_dataset)
    # st.test_image_file()
