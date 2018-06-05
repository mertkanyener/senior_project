import numpy as np
import cv2
import dlib
import os
import sys

from imutils import face_utils


# a simple function to create a rectangle around the face
def create_rect(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x  #width
    h = rect.bottom() - y  #height

    face_rect = (x, y, w, h)
    return face_rect


def face_landmarks(image_name):
    try:
        face_detector = dlib.get_frontal_face_detector()  # to detect the face shape
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # a trained algorithm to predict the face shape

        image = cv2.imread(image_name)

        faces = face_detector(image, 1)
        shape = 0 # giving a default value to prevent errors
        for face in faces:

            shape = shape_predictor(image, face)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = create_rect(face)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # put the face in a rectangle
        if type(shape) != int:  # if shape could not be found
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  # marking the face landmarks with red dots
            cv2.imwrite(image_name, image)
        else:
            print(image_name + "-- not processed")
    except TypeError:
        print("there's no such image")
        raise
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


# processing the directory
def landmarks_lib(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            image = os.path.join(subdir, file)
            print(file)
            face_landmarks(image)
    print("Images have been processed successfully.")


dir1 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]

#for n in dataset:
 #   landmarks_lib(n)





