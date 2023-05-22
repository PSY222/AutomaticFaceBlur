import cv2
import dlib
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from retinaface import RetinaFace


predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

def get_rects(image_path):
  faces = RetinaFace.detect_faces(image_path)
  if isinstance(faces, tuple):
    rects = None
  elif len(faces) != 0 and faces is not None:
    rects = [face_data["facial_area"] for face_data in faces.values()]
  else:
    rects = None

  return rects
  
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

def blur_paste(routes, img): 
    mask = np.zeros_like(img)  
    for landmarks in routes:
        mask = cv2.fillConvexPoly(mask, np.array(landmarks), (255, 255, 255))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  
    blurred_region = cv2.GaussianBlur(img, (51,51), 21) #Adjust the parameter to control the blur

    result_img = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    result_img = cv2.bitwise_or(result_img, cv2.bitwise_and(blurred_region, mask))

    return result_img

def blur_img(input_data):
    if isinstance(input_data, str):  # Check if input_data is a string (path)
        rects = get_rects(input_data)
        pil_img = Image.open(input_data).convert('RGB')
    else:  # Assuming input_data is a frame (image data)
        rects = get_rects(input_data)
        pil_img = Image.fromarray(input_data).convert('RGB')

    arr_img = np.asarray(pil_img)
    if not rects:
      return arr_img
      
    landmarks = get_landmarks(arr_img, rects)
    routes = get_faceline(landmarks)
    output = blur_paste(routes, arr_img)
    return output



def blur_video(path, output_path):
    capture = cv2.VideoCapture(path)

    if path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc,
                                 20.0, (int(capture.get(3)), int(capture.get(4))))

    frame_counter = 0
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    while True:
        _, frame = capture.read()
        frame_counter += 1

        if frame is None:
            print("frame is none")
            break

        # Perform frame processing here
        frame = blur_img(frame)

        if output_path:
            output.write(frame)
    print('Blurred video has been saved successfully at', output_path, 'path')
            
