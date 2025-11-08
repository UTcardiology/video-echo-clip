import os
import sys
import pandas as pd
import numpy as np
import webdataset as wds
import pydicom
from tqdm.auto import tqdm
import cv2
import tempfile
import json
import wandb
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from multiprocessing.managers import SyncManager


NUM_WORKERS = 64


class MyShardWriter(wds.ShardWriter):
    def __init__(self, pattern, maxcount=100000, maxsize=3e9, post=None, start_shard=0, **kw):
        super().__init__(pattern, maxcount, maxsize, post, start_shard)
        self.verbose = False

    def get_shards(self):
        return self.shard

    def get_count(self):
        return self.count if self.count < self.maxcount else 0

    def get_total(self):
        return self.total


class MyManager(SyncManager):
    pass


def worker(data, lock, pbar, sink):
    dicom_file_path, view_label, view_prob, report_text = data

    # NOTE: Eliminate views
    if view_label == "Other" or view_prob < 0.9:  # NOTE 4CH only -> if view_label != "4CH" or view_prob < 0.9:
        return

    # Videos
    dicom = pydicom.dcmread(dicom_file_path)
    ybr_frames = dicom.pixel_array  # (N, H, W, C), uint8, ybr color space
    # NOTE: Eliminate small frames
    # if len(ybr_frames) < 32:
    #     return
    bgr_frames = np.array([cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR) for img in ybr_frames])  # (N, H, W, C), uint8

    if hasattr(dicom, "FrameTime"):
        fps = round(1000 / float(dicom.FrameTime))
    else:
        fps = 30

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_filename = tmp.name

    # VideoWriter setting
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_filename, fourcc, fps, (bgr_frames.shape[2], bgr_frames.shape[1]))

    # Write frames
    for frame in bgr_frames:
        writer.write(frame)
    writer.release()

    # Read bytes and remove temp file
    with open(tmp_filename, "rb") as f:
        mp4_bytes = f.read()
    os.remove(tmp_filename)

    # Report texts
    report_text = report_text.split(",")

    # Info
    info_dic = {
        "report": report_text,
        "frame": bgr_frames.shape[0],
        "height": bgr_frames.shape[1],
        "width": bgr_frames.shape[2],
        "fps": fps,
        "dicom_path": dicom_file_path,
        "view_label": view_label,
        "view_prob": view_prob,
    }

    key_str = os.path.basename(dicom_file_path).split(".")[0]

    # Write tar file
    with lock:
        sink.write(
            {
                "__key__": key_str,
                "mp4": mp4_bytes,  # approx 1MB per dicom (same size)
                "json": json.dumps(info_dic),
            }
        )
        pbar.update(1)


def csv2wds(train_csv, val_csv, train_shards_dir, val_shards_dir):
    train_df = pd.read_csv(train_csv, sep=",")
    val_df = pd.read_csv(val_csv, sep=",")

    train_dicom_file_paths = train_df["dicom_path"]
    train_view_label = train_df["view_label"]
    train_view_prob = train_df["view_prob"]
    train_report_text = train_df["report_text"]

    val_dicom_file_paths = val_df["dicom_path"]
    val_view_label = val_df["view_label"]
    val_view_prob = val_df["view_prob"]
    val_report_text = val_df["report_text"]

    train_csv_data = [
        (
            train_dicom_file_paths[i],
            train_view_label[i],
            train_view_prob[i],
            train_report_text[i],
        )
        for i in range(len(train_dicom_file_paths))
    ]
    val_csv_data = [
        (
            val_dicom_file_paths[i],
            val_view_label[i],
            val_view_prob[i],
            val_report_text[i],
        )
        for i in range(len(val_dicom_file_paths))
    ]

    np.random.shuffle(train_csv_data)  # shuffle data before converting to webdataset
    np.random.shuffle(val_csv_data)  # shuffle data before converting to webdataset

    csv_data = [train_csv_data, val_csv_data]

    MyManager.register("Tqdm", tqdm)
    MyManager.register("Sink", MyShardWriter)

    print("Start!")
    sys.stdout.flush()

    shard_path = [train_shards_dir, val_shards_dir]
    for csv_idx in range(2):
        shard_dir_path = Path(shard_path[csv_idx])
        shard_dir_path.mkdir(exist_ok=True)
        shard_filename = str(shard_dir_path / "shards-%05d.tar")
        print("[Info] Shards are saved as", shard_filename)

        shard_size = int(50 * 1000**2)  # 50MB each

        with MyManager() as manager:
            lock = manager.Lock()
            pbar = manager.Tqdm(
                total=len(csv_data[csv_idx]),
                position=0,
            )
            pbar.set_description("Main process")
            sink = manager.Sink(
                pattern=shard_filename,
                maxsize=shard_size,
                maxcount=100,
            )

            worker_with_args = partial(
                worker,
                lock=lock,
                pbar=pbar,
                sink=sink,
            )
            with Pool(processes=NUM_WORKERS) as pool:
                pool.map(worker_with_args, csv_data[csv_idx], chunksize=len(csv_data[csv_idx]) // NUM_WORKERS)

            # Write json of dataset size
            dataset_size_filename = str(shard_dir_path / "dataset-size.json")
            with open(dataset_size_filename, "w") as fp:
                json.dump(
                    {
                        "dataset size": sink.get_total(),
                    },
                    fp,
                )

            sink.close()
            pbar.close()


if __name__ == "__main__":
    data_dir = os.path.expanduser("/path/to/csv/dir")

    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")

    train_shards_dir = os.path.join(data_dir, "train_shards")
    val_shards_dir = os.path.join(data_dir, "val_shards")

    np.random.seed(777)

    wandb.init(project="video-echo-clip", name="csv2wds")
    csv2wds(train_csv, val_csv, train_shards_dir, val_shards_dir)

    print("Complete!")
    wandb.finish()
