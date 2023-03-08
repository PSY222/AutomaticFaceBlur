import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from retinaface import RetinaFace
import dlib

def get_rects(image_path):
  faces = RetinaFace.detect_faces(image_path)
  rects = [faces[f"face_{i+1}"]['facial_area'] for i in range(len(faces))]
  return rects

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image, rects):
    landmarks = []
    for rect in rects:
        x1, y1, x2, y2 = map(int, map(round, rect))
        rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
        shape = predictor(image, rect)
        landmarks.append(np.array([[p.x, p.y] for p in shape.parts()]))
    return landmarks

def get_faceline(landmarks):
    routes = []
    faces = []
    for p in range(len(landmarks)):
        for i in range(15, -1, -1):
            from_coordinate = landmarks[p][i+1]
            to_coordinate = landmarks[p][i]
            faces.append(from_coordinate)
        from_coordinate = landmarks[p][0]
        to_coordinate = landmarks[p][17]
        faces.append(from_coordinate)
        
        for i in range(17, 20):
            from_coordinate = landmarks[p][i]
            to_coordinate = landmarks[p][i+1]
            faces.append(from_coordinate)
        
        from_coordinate = landmarks[p][19]
        to_coordinate = landmarks[p][24]
        faces.append(from_coordinate)
        
        for i in range(24, 26):
            from_coordinate = landmarks[p][i]
            to_coordinate = landmarks[p][i+1]
            faces.append(from_coordinate)
        
        from_coordinate = landmarks[p][26]
        to_coordinate = landmarks[p][16]
        faces.append(from_coordinate)
        faces.append(to_coordinate)
        routes.append(faces)
        faces = []
    return routes

def blur_img(routes,img,blur_size=21,sigmax=3):
    for landmarks in routes:
        mask = np.zeros((img.shape[0], img.shape[1]),np.uint8)
        mask = cv2.fillConvexPoly(mask, np.array(landmarks), (255, 255, 255))
        blurred_image = cv2.GaussianBlur(img, (blur_size, blur_size), sigmax)
        m = cv2.moments(mask)
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
        center = (cX, cY)
        img = cv2.seamlessClone(blurred_image, img, mask, center, cv2.NORMAL_CLONE)
    return img