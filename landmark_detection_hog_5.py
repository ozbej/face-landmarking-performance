from imutils import face_utils
import dlib
import cv2
import os
from scipy.spatial import distance
import numpy as np
from math import sqrt

def get_landmarks_gt(path):
    landmarks_gt = list()
    f = open(path, "r")
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i < 3 or i == len(lines)-1: continue
        x_gt, y_gt = line.split()
        landmarks_gt.append(np.array([float(x_gt), float(y_gt)]))
    return np.array(landmarks_gt)


def get_annotated_face(rects, landmarks_gt):
    rect_dict = dict()
    for i, rect in enumerate(rects):
        c = 0
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        for (x_gt, y_gt) in landmarks_gt:
            if (x <= x_gt <= x + w or x >= x_gt >= x + w) and (y <= y_gt <= y + h or y >= y_gt >= y + h):
                c += 1
        rect_dict[i] = c
    max_key = max(rect_dict, key=rect_dict.get)
    if rect_dict[max_key] / 5 < 0.5:
        return rects[max_key], True
    return rects[max_key], False


def get_iod(landmarks):
    # Left eye center
    x_left = landmarks[2][0] + landmarks[3][0]
    y_left = landmarks[2][1] + landmarks[3][1]
    centroid_left = (int(x_left / 2), int(y_left / 2))
    # Right eye center
    x_right = landmarks[0][0] + landmarks[1][0]
    y_right = landmarks[0][1] + landmarks[1][1]
    centroid_right = (int(x_right / 2), int(y_right / 2))
    # IOD
    return distance.euclidean(centroid_left, centroid_right)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# Set input and output dir
input_dir = "./data/ibug/ibug_masked/"
output_dir = "./data_landmarked/dlib_5/ibug/ibug_masked_hog/"

nrmse_global = list()
count_incorrect = 0
count_all = 0

for count, filename in enumerate(os.listdir(input_dir)):
    if not (filename.__contains__(".jpg") or filename.__contains__(".png")): continue
    error = False
    count_all += 1
    print("Processing: %s (%d)" % (filename, count_all))

    filename_s = os.path.splitext(filename)[0]
    landmarks_5 = [45, 42, 36, 39, 33]
    landmarks_gt = get_landmarks_gt(input_dir+filename_s+".pts")
    landmarks_gt = landmarks_gt[landmarks_5]

    image = cv2.imread(input_dir+filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Draw all ground truth landmarks
    for i, (x, y) in enumerate(landmarks_gt):
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Get faces
    rects = detector(gray, 0)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    if len(rects) == 0:
        cv2.imwrite(output_dir+"errors/"+filename, image)
        count_incorrect += 1
        continue
    else:
        face, error = get_annotated_face(rects, landmarks_gt)
        if error:
            count_incorrect += 1

    # Make the prediction and transform it to numpy array
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)

    # IOD
    iod = get_iod(landmarks)

    distances = list()
    # Draw all predicted landmarks
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        distance_normalized = (distance.euclidean(landmarks[i], landmarks_gt[i]) / iod)**2
        distances.append(distance_normalized)
    nrmse_local = sqrt(sum(distances)/len(distances))

    if error:
        cv2.imwrite(output_dir + "errors/" + filename, image)
    else:
        nrmse_global.append(nrmse_local)
        cv2.imwrite(output_dir + filename, image)

print("NRMSE: %.5f, skipped %d of %d (%.3f)" % (np.mean(np.array(nrmse_global)), count_incorrect, count_all, count_incorrect/count_all))
