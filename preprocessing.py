import numpy as np
import cv2
import dlib
import os

from imutils import face_utils
from aligner import FaceAligner


class Preprocessing:

    def __init__(self, dataset=None):
        self.dataset = dataset

    # Get the average width and height of cut photos
    @staticmethod
    def get_avg_size(dataset):
        x_values = []
        y_values = []
        for group in dataset:
            for subdir, dirs, files in os.walk(group):
                for file in files:
                    img_name = os.path.join(subdir, file)
                    img = cv2.imread(img_name)
                    x_values.append(img.shape[1])
                    y_values.append(img.shape[0])
        return int(np.mean(x_values)), int(np.mean(y_values))

    # Make alignment on photos in dataset
    @staticmethod
    def make_alignment(img_name):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        img = cv2.imread(img_name, 0)
        rectangles = detector(img, 1)
        rect = rectangles[0]
        fa = FaceAligner(predictor, desiredFaceWidth=900, desiredFaceHeight=677)
        faceAligned = fa.align(img, rect)
        faceAligned = faceAligned[24:480, 173:811]
        cv2.imwrite(img_name, faceAligned)
    @staticmethod
    def align_all(dataset):

        prp = Preprocessing()

        for group in dataset:
            for subdir, dirs, files in os.walk(group):
                for file in files:
                    image_name = os.path.join(subdir, file)
                    prp.make_alignment(image_name)

    @staticmethod
    def vectorize_age_old(dataset):
        X_age = []
        X_forehead = []
        X_reye_side = []
        X_leye_side = []
        X_reye_under = []
        X_leye_under = []
        X_mouth_right = []
        X_mouth_left = []
        y_age = []
        count = 0

        forehead_xvals = []
        forehead_yvals = []

        reye_side_xvals = []
        reye_side_yvals = []

        leye_side_xvals = []
        leye_side_yvals = []

        reye_under_xvals = []
        reye_under_yvals = []

        leye_under_xvals = []
        leye_under_yvals = []

        rmouth_xvals = []
        rmouth_yvals = []

        lmouth_xvals = []
        lmouth_yvals = []

        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        for group in dataset:
            count += 1
            for subdir, dirs, files in os.walk(group):
                for file in files:
                    img_name = os.path.join(subdir, file)
                    img = cv2.imread(img_name)

                    rectangles = face_detector(img, 1)

                    shape = shape_predictor(img, rectangles[0])
                    shape = face_utils.shape_to_np(shape)

                    y1 = shape[25]
                    y2 = shape[25]
                    x1 = shape[18]
                    x2 = shape[16]
                    forehead = img[(y1[1] - 45):(y2[1] - 5), x1[0]:x2[0]]
                    forehead_xvals.append(forehead.shape[1])
                    forehead_yvals.append(forehead.shape[0])
                    X_forehead.append(forehead)

                    y1 = shape[18]
                    y2 = shape[30]
                    x1 = shape[1]
                    x2 = shape[37]
                    reye_side = img[y1[1]:y2[1], x1[0]:x2[0]]
                    reye_side_xvals.append(reye_side.shape[1])
                    reye_side_yvals.append(reye_side.shape[0])
                    X_reye_side.append(reye_side)

                    y1 = shape[27]
                    y2 = shape[30]
                    x1 = shape[46]
                    x2 = shape[16]
                    leye_side = img[y1[1]:y2[1], x1[0]:x2[0]]

                    leye_side_xvals.append(leye_side.shape[1])
                    leye_side_yvals.append(leye_side.shape[0])
                    X_leye_side.append(leye_side)

                    y1 = shape[42]
                    y2 = shape[30]
                    x1 = shape[37]
                    x2 = shape[40]
                    reye_under = img[y1[1]:y2[1], x1[0]:x2[0]]
                    reye_under_xvals.append(reye_under.shape[1])
                    reye_under_yvals.append(reye_under.shape[0])
                    X_reye_under.append(reye_under)

                    y1 = shape[47]
                    y2 = shape[30]
                    x1 = shape[43]
                    x2 = shape[46]
                    leye_under = img[y1[1]:y2[1], x1[0]:x2[0]]
                    leye_under_xvals.append(leye_under.shape[1])
                    leye_under_yvals.append(leye_under.shape[0])
                    X_leye_under.append(leye_under)

                    y1 = shape[34]
                    y2 = shape[6]
                    x1 = shape[6]
                    x2 = shape[49]
                    mouth_right = img[y1[1]:y2[1], x1[0]:x2[0]]
                    rmouth_xvals.append(mouth_right.shape[1])
                    rmouth_yvals.append(mouth_right.shape[0])
                    X_mouth_right.append(mouth_right)

                    y1 = shape[34]
                    y2 = shape[12]
                    x1 = shape[55]
                    x2 = shape[12]
                    mouth_left = img[y1[1]:y2[1], x1[0]:x2[0]]
                    lmouth_xvals.append(mouth_left.shape[1])
                    lmouth_yvals.append(mouth_left.shape[0])
                    X_mouth_left.append(mouth_left)

                    if count == 1:
                        y_age.append(1)
                    elif count == 2:
                        y_age.append(1)
                    elif count == 3:
                        y_age.append(2)
                    elif count == 4:
                        y_age.append(2)
                    elif count == 5:
                        y_age.append(3)
                    elif count == 6:
                        y_age.append(3)
                    elif count == 7:
                        y_age.append(4)
                    elif count == 8:
                        y_age.append(4)

        xfore_mean, yfore_mean = int(np.mean(forehead_xvals)), int(np.mean(forehead_yvals))
        xrside_mean, yrside_mean = int(np.mean(reye_side_xvals)), int(np.mean(reye_side_yvals))

        xlside_mean, ylside_mean = int(np.mean(leye_side_xvals)), int(np.mean(leye_side_yvals))
        xrunder_mean, yrunder_mean = int(np.mean(reye_under_xvals)), int(np.mean(reye_under_yvals))
        xlunder_mean, ylunder_mean = int(np.mean(leye_under_xvals)), int(np.mean(leye_under_yvals))
        xrmouth_mean, yrmouth_mean = int(np.mean(rmouth_xvals)), int(np.mean(rmouth_yvals))
        xlmouth_mean, ylmouth_mean = int(np.mean(lmouth_xvals)), int(np.mean(lmouth_yvals))

        for i in range(len(X_reye_side)):
            X_forehead[i] = cv2.resize(X_forehead[i], (xfore_mean, yfore_mean))
            X_forehead[i] = np.matrix.flatten(X_forehead[i])

            X_reye_side[i] = cv2.resize(X_reye_side[i], (xrside_mean, yrside_mean))
            X_reye_side[i] = np.matrix.flatten(X_reye_side[i])

            X_leye_side[i] = cv2.resize(X_leye_side[i], (xlside_mean, ylside_mean))
            X_leye_side[i] = np.matrix.flatten(X_leye_side[i])
            X_reye_under[i] = cv2.resize(X_reye_under[i], (xrunder_mean, yrunder_mean))
            X_reye_under[i] = np.matrix.flatten(X_reye_under[i])
            X_leye_under[i] = cv2.resize(X_leye_under[i], (xlunder_mean, ylunder_mean))
            X_leye_under[i] = np.matrix.flatten(X_leye_under[i])
            X_mouth_right[i] = cv2.resize(X_mouth_right[i], (xrmouth_mean, yrmouth_mean))
            X_mouth_right[i] = np.matrix.flatten(X_mouth_right[i])
            X_mouth_left[i] = cv2.resize(X_mouth_left[i], (xlmouth_mean, ylmouth_mean))
            X_mouth_left[i] = np.matrix.flatten(X_mouth_left[i])
            X_age.append(np.concatenate((X_forehead[i], X_reye_side[i], X_leye_side[i],
                                         X_reye_under[i], X_leye_under[i], X_mouth_right[i], X_mouth_left[i])))

        return X_age, y_age


    # Read data from path and save photos as matrixes in X.
    # Save class labels in y_age and y_gender
    @staticmethod
    def read_data(dataset):

        X = []
        y_age = []
        y_gender = []

        prp = Preprocessing()
        #x_mean, y_mean = prp.get_avg_size(dataset)
        count = 0

        for group in dataset:
            count += 1
            for subdir, dirs, files in os.walk(group):
                for file in files:
                    img_name = os.path.join(subdir, file)
                    img = cv2.imread(img_name, 0)
                    img = cv2.resize(img, (293, 293))
                    X.append(img)
                    if count == 1:
                        y_gender.append(0)
                        y_age.append(1)
                    elif count == 2:
                        y_gender.append(1)
                        y_age.append(1)
                    elif count == 3:
                        y_gender.append(0)
                        y_age.append(2)
                    elif count == 4:
                        y_gender.append(1)
                        y_age.append(2)
                    elif count == 5:
                        y_gender.append(0)
                        y_age.append(3)
                    elif count == 6:
                        y_gender.append(1)
                        y_age.append(3)
                    elif count == 7:
                        y_gender.append(0)
                        y_age.append(4)
                    elif count == 8:
                        y_gender.append(1)
                        y_age.append(4)

        return X, y_gender, y_age

    # Vectorize all images in X
    @staticmethod
    def vectorize(X):

        for i in range(len(X)):
            X[i] = np.matrix.flatten(X[i])

        return X


"""

dir1 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]

prp = Preprocessing()
prp.align_all(dataset)

"""





