import torchvision.transforms as transforms
import cv2
import csv
import dlib
import numpy as np
import math
from scipy.spatial.distance import euclidean
import json
import os
import statistics
import torch
from torchvision import models
from retinaface import RetinaFace
import random

vgg16 = models.vgg16(pretrained=True)
vgg16_features = models.vgg16(pretrained=True).features
predictor_path = r"shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
image_folder_path = r".\faces_to_extract_from"
landmark_coords = []

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)


def biggest(list1, el):
    try:
        list1.remove(el)
    except Exception as e:
        print("Invalid input")
        return False
    for i in list1:
        if el <= i:
            return False
    return True

def similar(a, b):
    diff = 0.1
    return math.isclose(a, b, abs_tol=diff)
    
def all_similar4(a, b, c, d):
    val = False
    list_m = [a,b,c,d]
    for i in range(len(list_m) - 1):
        if similar(a, list_m[i+1]):
            val = True
        else:
            return False
    if (val == True) and (similar(b, c)) and (similar(b,d) and similar(c,d)):
        return True
    else:
        return False
            
def all_similar3(a, b, c):
    val = False
    list_m = [a,b,c]
    for i in range(len(list_m) - 1):
        if similar(a, list_m[i+1]):
            val = True
        else:
            return False
    if (val == True) and (similar(b, c)):
        return True
    else:
        return False

def give_tensor(input_tens):
    vgg16.eval()
    with torch.no_grad():
        output = vgg16_features(input_tens)
    print(output)

for filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {filename}")
        continue

    faces = RetinaFace.detect_faces(image)
    if len(faces) == 0:
        print(f"No faces detected in {filename}.")
        continue

    for key in faces.keys():
        face_info = faces[key]
        facial_area = face_info["facial_area"]

        x1, y1, x2, y2 = facial_area
        originaly1 = facial_area[1]
        paddingx = 50
        paddingy = 80
        x1 = max(0, x1 - paddingx)
        y1 = max(0, y1 - paddingy)
        x2 = min(image.shape[1], x2 + paddingx)
        y2 = min(image.shape[0], y2 + paddingy)

        face_crop = image[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop_resized = cv2.resize(face_crop_rgb, (224, 224))

        face_tensor_transformer = transforms.ToTensor()
        face_tensor = face_tensor_transformer(face_crop_resized)
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = normalize_transform(face_tensor)
        input_tensor = input_tensor.unsqueeze(0)

        face_rect = dlib.rectangle(0, 0, face_crop.shape[1], face_crop.shape[0])
        landmarks = predictor(face_gray, face_rect)

        landmark_coords = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_coords.append((x, y))
            cv2.circle(face_crop, (x, y), 2, (0, 255, 0), -1)
            
        chin = landmark_coords[8]
        nose_bridge = landmark_coords[27]

        eyebrow_y = (landmark_coords[21][1] + landmark_coords[22][1]) // 2
        nose_bridge_y = landmark_coords[27][1]
        forehead_proxy = eyebrow_y - nose_bridge_y
        scale_factor = 5.55 / forehead_proxy if forehead_proxy != 0 else 1
        hairline_y = int(eyebrow_y - (5.55 / scale_factor))

        face_height = abs(landmark_coords[8][1] - hairline_y)

        face_width = euclidean(landmark_coords[1], landmark_coords[15])


        jaw_width = euclidean(landmark_coords[4], landmark_coords[12])

        normalized_jaw_width = jaw_width / face_width


        left_jaw_angle = calculate_angle(landmark_coords[2],
                                         landmark_coords[3],
                                         landmark_coords[4])

        right_jaw_angle = calculate_angle(landmark_coords[14],
                                          landmark_coords[13],
                                          landmark_coords[12])

        left_eye_center = ((landmark_coords[36][0] + landmark_coords[39][0])/2, 
                           (landmark_coords[36][1] + landmark_coords[39][1])/2)
        right_eye_center = ((landmark_coords[42][0] + landmark_coords[45][0])/2, 
                            (landmark_coords[42][1] + landmark_coords[45][1])/2)
        symmetry_score = euclidean(left_eye_center, right_eye_center) / face_width

        upper_head = euclidean(landmark_coords[1], landmark_coords[17])
        cheekbone = euclidean(landmark_coords[2], landmark_coords[16])
        jaw = euclidean(landmark_coords[5], landmark_coords[13])
        facelength = face_height
        json_file = "extracted_data.json"
        data = []
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
        
        print(filename)
        with open(r".\outputs\extracted_data.txt", "a") as f:
            f.write(f"Name of the file: {filename}\n")
            f.write(f"Normalized jaw width: {normalized_jaw_width:.2f} (average: 0.6-0.7)\n[[2]]")
            f.write(f"Jaw angles: Left={left_jaw_angle:.1f}, Right={right_jaw_angle:.1f}\n (average: 160-180)")
            f.write(f"Eye Symmetry score: {symmetry_score:.2f} (out of 1.0)\n[[8]]")
            print("extracted")