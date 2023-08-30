import os
from icrawler.builtin import GoogleImageCrawler
import pandas as pd
import time

# paths
IMAGE_DIR = "./images"
NAME_INDEX = "./name_index.csv"
DATASET = "../dataset_12"

if __name__ == "__main__":
    # create image directory (if not exist)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # define google_crawler
    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={"root_dir": IMAGE_DIR},
    )
    filters = dict(
        size="large",
    )

    # read name_index.csv
    df = pd.read_csv(NAME_INDEX)
    # create is_crawled column if first run
    # df['is_crawled'] = False

    # clear (re-empty) IMAGE_DIR folder (if not empty)
    os.system(f"rm -rf {IMAGE_DIR}/*")

    for i, row in df.iterrows():
        name = row["name"]
        keyword = row["keyword"]
        print(f"\n{i} - crawling {name}...\n")
        if row["is_crawled"]:
            print(f"{name} is already crawled!")
            continue
        google_crawler.crawl(
            keyword=keyword,
            max_num=1000,
            filters=filters,
            file_idx_offset="auto",
        )

        # create directory for name and move images to this directory
        dir_name = str(i)  # replace space with underscore and lowercase
        os.makedirs(os.path.join(DATASET, dir_name), exist_ok=True)
        # mv all image with any type except dirs to dir_name
        os.system(f"mv {IMAGE_DIR}/* {DATASET}/{dir_name}")

        # update is_crawled
        df.loc[i, "is_crawled"] = True
        # save name_index.csv
        df.to_csv(NAME_INDEX, index=False)
        print(f"\ncrawled {name}!\n")

        # sleep 1 sec
        print("sleep 1 sec...")
        time.sleep(1)
