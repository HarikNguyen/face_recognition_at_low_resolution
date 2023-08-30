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

for i, name in enumerate(sorted(os.listdir(DATASET))):
    path_to_save = os.path.join(IMAGE_DIR, str(name))
    # if name in name_index not exist, create it
    if not df.loc[i, "detected"]:
        # create folder or remove all files in folder
        os.makedirs(path_to_save, exist_ok=True)
        # remove all files in folder
        for file in os.listdir(path_to_save):
            os.remove(os.path.join(path_to_save, file))

    print(f"Detecting and cropping {name}...")
    counter = 0
    for j, img_name in enumerate(sorted(os.listdir(os.path.join(DATASET, name)))):
        in_img_path = os.path.join(DATASET, name, img_name)
        print(f"{j} - {in_img_path}")

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
            out_img_path = os.path.join(path_to_save, f"{counter}.jpg")
            x1, y1, x2, y2 = box
            face = img[int(y1) : int(y2), int(x1) : int(x2)].copy()
            # convert to BGR
            try:
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_img_path, face)
                print(f"saved {img_name}!")
            except:
                counter -= 1
            # release memory
            del face
            torch.cuda.empty_cache()
        del img

    # update detected column
    df.loc[i, "detected"] = True
    # save name_index.csv
    df.to_csv(NAME_INDEX, index=False)
    print(f"Detected and cropped {name}!\n")
