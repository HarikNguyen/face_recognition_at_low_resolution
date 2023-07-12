# Statistic images in each individual directory

import os
import pandas as pd

NAME_INDEX = "./name_index.csv"
DATASET = "../dataset"

if __name__ == "__main__":
    # load name_index.csv
    df = pd.read_csv(NAME_INDEX)

    # create scrawled_count column if first run
    df["scrawled_count"] = 0

    # for each individual directory (named by id) in dataset directory
    # count number of images in it and update scrawled_count
    for dir_name in os.listdir(DATASET):
        if not os.path.isdir(os.path.join(DATASET, dir_name)):
            continue
        scrawled_count = len(os.listdir(os.path.join(DATASET, dir_name)))
        df.loc[int(dir_name), "scrawled_count"] = scrawled_count

    # convert scrawled_count to int
    df["scrawled_count"] = df["scrawled_count"].astype(int)

    # print sum of scrawled_count
    print(f"Total scrawled images: {df['scrawled_count'].sum()}")

    # save name_index.csv
    df.to_csv(NAME_INDEX, index=False)
