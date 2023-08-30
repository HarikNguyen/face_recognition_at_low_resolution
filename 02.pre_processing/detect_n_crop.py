import threading
import os
import pandas as pd
import cv2
import torch
from facenet_pytorch import MTCNN

DATASET = "../dataset_12"
NAME_INDEX = "./name_index.csv"
IMAGE_DIR = "../images_12"

os.makedirs(IMAGE_DIR, exist_ok=True)

detected_list = []

# load name_index.csv
df = pd.read_csv(NAME_INDEX)
# if detected column not exist, create it
if "detected" not in df.columns:
    df["detected"] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Detect with {device}...\n")

mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)


def detect_n_crop(dir_in, dir_path_out):
    counter = 0
    for i, img_name in enumerate(sorted(os.listdir(dir_in))):
        in_img_path = os.path.join(dir_in, img_name)
        print(f"{i} - {in_img_path}")

        # load image
        img = cv2.imread(in_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect face
        boxes, probs = mtcnn.detect(img)

        # crop face
        if boxes is None:
            print(f"{img_name} has no face!")
            continue
        for box in boxes:
            counter += 1
            out_img_path = os.path.join(dir_path_out, f"{counter}.jpg")
            x1, y1, x2, y2 = box
            face = img[int(y1) : int(y2), int(x1) : int(x2)].copy()
            # convert to BGR
            try:
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                # resize to 250x250
                face = cv2.resize(face, (250, 250))
                cv2.imwrite(out_img_path, face)
                print(f"saved {img_name}!")
            except:
                counter -= 1
            # release memory
            del face
            torch.cuda.empty_cache()
        del img

    print(f"Detected and cropped {dir_in}!\n")


class DetectNCropThreading(threading.Thread):
    def __init__(self, dataset_part):
        threading.Thread.__init__(self)
        self.dataset_part = dataset_part

    def run(self):
        for name_index in self.dataset_part:
            dir_in = os.path.join(DATASET, str(name_index))
            dir_path_out = os.path.join(IMAGE_DIR, str(name_index))
            if not df.loc[int(name_index), "detected"]:
                # create folder or remove all files in folder
                os.makedirs(dir_path_out, exist_ok=True)
                # remove all files in folder
                for file in os.listdir(dir_path_out):
                    os.remove(os.path.join(dir_path_out, file))
            print(f"Detecting and cropping {name_index}...")
            detect_n_crop(dir_in, dir_path_out)
            # update detected column
            df.loc[int(name_index), "detected"] = True
            # save name_index.csv
            df.to_csv(NAME_INDEX, index=False)
            print(f"Detected and cropped {name_index}!\n")


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
