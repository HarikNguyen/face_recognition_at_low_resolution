import os
import cv2
import json
import threading
import pandas as pd

NAME_INDEX = "./name_index.csv"
FACE_LIST_FILE = "./face_list.json"
DATASET_DIR = "../dataset_4"
IMAGE_DIR = "../images"

# get face_list (detected faces)
face_list = {}
with open(FACE_LIST_FILE, "r") as f:
    face_list = json.load(f)

# load name_index.csv
df = pd.read_csv(NAME_INDEX)
# if detected column not exist, create it
if "detected" not in df.columns:
    df["detected"] = False


def crop(dir_in, dir_out, name_index):
    output_img_index = 0
    # get face_list of name_index
    img_dict_by_name_index = face_list[name_index]
    for img_name in img_dict_by_name_index.keys():
        # get face_list of img_name
        face_list_of_img = img_dict_by_name_index[img_name]
        # define path to input image (img_path_in)
        img_path_in = os.path.join(dir_in, img_name)
        # read image
        img = cv2.imread(img_path_in)
        # crop and save detected faces
        for face_box in face_list_of_img:
            # update output_img_index
            output_img_index += 1
            x1, y1, x2, y2 = face_box
            face = img[int(y1) : int(y2), int(x1) : int(x2)].copy()
            # define path to output image (img_path_out)
            img_path_out = os.path.join(dir_out, f"{output_img_index}.jpg")
            # crop and save image
            try:
                # face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                # resize to 250x250
                face = cv2.resize(face, (250, 250))
                cv2.imwrite(img_path_out, face)
            except:
                output_img_index -= 1
                continue
            # release memory
            del face
        del img


class CropThreading(threading.Thread):
    def __init__(self, dataset_part):
        threading.Thread.__init__(self)
        self.dataset_part = dataset_part

    def run(self):
        for name_index in self.dataset_part:
            # define path to input dir (dir_in)
            dir_in = os.path.join(DATASET_DIR, str(name_index))
            # define path to output dir (dir_path_out)
            dir_path_out = os.path.join(IMAGE_DIR, str(name_index))
            # check if name_index is detected, if not, create folder
            # or remove all files in folder. Otherwise, continue
            if not df.loc[int(name_index), "detected"]:
                # create folder or remove all files in folder
                os.makedirs(dir_path_out, exist_ok=True)
                # remove all files in folder
                for file in os.listdir(dir_path_out):
                    os.remove(os.path.join(dir_path_out, file))
            else:
                continue
            # crop images in dir_in and save to dir_path_out
            print(f"Cropping {name_index}...")
            crop(dir_in, dir_path_out, name_index)
            # update detected column
            df.loc[int(name_index), "detected"] = True
            # save name_index.csv
            df.to_csv(NAME_INDEX, index=False)
            print(f"Cropped {name_index}!\n")


num_thread = 4
dataset_list = sorted(os.listdir(DATASET_DIR))
# split dataset_list into num_thread parts
dataset_parts = [dataset_list[i::num_thread] for i in range(num_thread)]

# create threads to detect and crop
detected_list = []
for i in range(num_thread):
    thread = CropThreading(dataset_parts[i])
    thread.start()
    detected_list.append(thread)

# wait for all threads to complete
for thread in detected_list:
    thread.join()
