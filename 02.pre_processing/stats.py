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

    # print time of last update
    print(f"Last updated: {time.ctime(os.path.getmtime(NAME_INDEX))}")
    # print sum of detected_count
    print(f"Total detected images: {df['detected_count'].sum()}/{len(df)}")
    # print individual was detected
    print(f"Total detected individuals: {len(df[df['detected'] == True])}")

    # get size of dataset dir
    size = get_dir_size(DATASET) / 1024 / 1024 / 1024
    print(f"Total size of dataset: ~{size:.2f} GB")

    print(df.loc[124, :])


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
