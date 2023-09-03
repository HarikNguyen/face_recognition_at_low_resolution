import os
import pandas as pd
import argparse
import time

# Path to name_index.csv
NAME_INDEX = "./name_index.csv"


def get_dir_size(path):
    total_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size


def stats(config):
    DATASET = config.dataset

    # load name_index.csv
    df = pd.read_csv(NAME_INDEX)
    # initialize detected_count column if not exist
    if "detected_count" not in df.columns:
        df["detected_count"] = 0

    # for each individual directory (named by id) in dataset directory
    # count number of images in it and update detected_count
    for dir_name in os.listdir(DATASET):
        if not os.path.isdir(os.path.join(DATASET, dir_name)):
            continue
        detected_count = len(os.listdir(os.path.join(DATASET, dir_name)))
        df.loc[int(dir_name), "detected_count"] = detected_count

    # convert detected_count to int
    df["detected_count"] = df["detected_count"].astype(int)

    # update name_index.csv file
    df.to_csv(NAME_INDEX, index=False)

    print("\n\t\t\033[01m\033[37m Statistics\033[37m\033[00m\n")
    # print time of last update
    print(
        f"\033[96m Last updated:\033[00m \033[95m{time.ctime(os.path.getmtime(NAME_INDEX))}\033[00m"
    )
    # print sum of detected_count
    print(
        f"\033[96m Total detected images::\033[00m \033[95m{df['detected_count'].sum()}\033[00m"
    )
    # print individual was detected
    detected_num = len(df[df["detected"] == True])
    total_num = len(df)
    percent = detected_num / total_num * 100
    print(
        f"\033[96m Total detected individuals:\033[00m \033[95m{detected_num}/{total_num} \033[00m|| \033[92m{percent:.2f}%\033[00m"
    )

    # get size of dataset dir
    size = get_dir_size(DATASET) / 1024 / 1024 / 1024
    print(f"\033[96m Total size of dataset:\033[00m \033[95m~{size:.2f} GB\033[00m")


if __name__ == "__main__":
    # define argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--dataset", type=str, default="../images", help="Path to dataset directory"
    )

    # parse arguments
    config = parser.parse_args()

    # call stats function
    stats(config)
