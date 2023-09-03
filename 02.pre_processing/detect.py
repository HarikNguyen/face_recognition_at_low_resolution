import os
import json
import cv2
import threading
import pandas as pd
import torch
from facenet_pytorch import MTCNN

DATASET = "../dataset_4"

face_list = {}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Detect {DATASET.replace('..','').replace('/','')} with {device}...\n")

mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

# check if face_list.json not exists, create it
if not os.path.exists("face_list.json"):
    with open("face_list.json", "w") as f:
        json.dump(face_list, f)


def detect_n_crop(dir_in, name_index):
    face_list = {}
    # read face_list.json
    try:
        with open("face_list.json", "r") as f:
            face_list = json.load(f)
            f.close()
    except:
        print("Error in reading face_list.json")
        return
    # check if name_index is in it
    if str(name_index) in face_list.keys():
        return
    # if not, detect and update face_list.json
    img_face_dict = {}
    for img_name in sorted(os.listdir(dir_in)):
        in_img_path = os.path.join(dir_in, img_name)
        # load image
        img = cv2.imread(in_img_path)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            continue
        # detect face
        boxes, _ = mtcnn.detect(img)

        # crop face
        if boxes is None:
            continue
        img_face_dict[img_name] = boxes.tolist()
        torch.cuda.empty_cache()
        del img
    face_list[name_index] = img_face_dict
    print(f"Detected and cropped {dir_in}!\n")
    try:
        # update face_list.json
        with open("face_list.json", "w") as f:
            json.dump(face_list, f)
            f.close()
    except:
        print("Error in updating face_list.json")
        return


class DetectNCropThreading(threading.Thread):
    def __init__(self, dataset_part):
        threading.Thread.__init__(self)
        self.dataset_part = dataset_part

    def run(self):
        for name_index in self.dataset_part:
            dir_in = os.path.join(DATASET, str(name_index))
            print(f"Detecting and cropping {name_index}...")
            detect_n_crop(dir_in, int(name_index))


num_thread = 4
dataset_list = sorted(os.listdir(DATASET))
# split dataset_list into num_thread parts
dataset_parts = [dataset_list[i::num_thread] for i in range(num_thread)]

# create threads to detect and crop
detected_list = []
for i in range(num_thread):
    thread = DetectNCropThreading(dataset_parts[i])
    thread.start()
    detected_list.append(thread)

# wait for all threads to complete
for thread in detected_list:
    thread.join()
