import cv2
import dlib
import os
import numpy as np
import facial_landmark as fl

from imutils import face_utils


# This function cuts the face from images

def cut_face(image_name):


    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(image_name)

    rectangles = face_detector(image, 1)
    for (i, rectangle) in enumerate(rectangles):
        shape = shape_predictor(image, rectangle)
        shape = face_utils.shape_to_np(shape)
        x = shape[1]
        y = shape[25]
        w = shape[16]
        h = shape[9]
        #cv2.rectangle(image, (x[0], y[1] - 20), (w[0], h[1] + 10), (0, 255,0), 2)
    face = image[(y[1]-60):(h[1]+10), (x[0] - 10):(w[0] + 10)]

    #cv2.imshow('img', face)
    cv2.imwrite(image_name ,face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def landmarks_lib(dataset):
    for group in dataset:
        for subdir, dirs, files in os.walk(group):
            for file in files:
                image = os.path.join(subdir, file)
                print(file)
                cut_face(image)
        print("Images have been processed successfully.")


dir1 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]


landmarks_lib(dataset)


