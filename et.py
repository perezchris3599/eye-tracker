import cv2
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in slide]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)

    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')
left = [36, 37, 38, 39, 40, 41]
ight = [42, 43, 44, 45, 46, 47]
cap = cv2.VideoCapture(0)
ret, img = cap.read()