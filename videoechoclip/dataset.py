import os
import io
import logging
import math
from dataclasses import dataclass
from multiprocessing import Value
import av
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pydicom
import cv2
from functools import partial
import webdataset as wds
import json
import warnings

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*")


def get_data(args, preprocess_train, preprocess_val, epoch=0, tokenizer=None):
    data = {}
    if args.train_data:
        data["train"] = get_dataset_fn(args.dataset_type)(args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
    if args.val_data:
        data["val"] = get_dataset_fn(args.dataset_type)(args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data


def get_dataset_fn(dataset_type):
    if dataset_type == "webdataset":
        return _build_wds_dataset
    elif dataset_type == "csv":
        return _build_csv_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def _build_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    Dataset using CSV file.
    """
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        num_frames=args.model.vision.num_frames,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class CsvDataset(Dataset):
    """
    CSV dataset for echocardiography videos and reports.
    """

    def __init__(self, input_filename, transforms, tokenizer=None, num_frames=32):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=",")

        # Filter dicom file (exclude views and low probs)
        # df = df[(df["view_prob"] > 0.9) & (df["view_label"] == "4CH")]  # NOTE training with 4CH only
        df = df[(df["view_prob"] > 0.9) & (df["view_label"] != "Other")]  # NOTE training with multi-view

        self.dicom_path = df["dicom_path"]
        self.report_text = df["report_text"]
        self.view_label = df["view_label"]
        self.view_prob = df["view_prob"]

        self.transforms = transforms
        logging.debug("Done loading data.")

        self.tokenizer = tokenizer

        self.num_frames = num_frames

    def __len__(self):
        return len(self.dicom_path)

    def __getitem__(self, idx):
        dicom = pydicom.dcmread(self.dicom_path.iloc[idx])

        # frames preprocess
        stride = 2  # NOTE stride 2
        ybr_frames = dicom.pixel_array[: self.num_frames * stride : stride]  # (N, H, W, C), uint8, ybr color space
        rgb_frames = np.array([cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB) for img in ybr_frames])  # (N, H, W, C), uint8

        if len(rgb_frames) < self.num_frames:
            # frames is shorter than num_frames
            rgb_frames = np.concatenate([rgb_frames, np.array([rgb_frames[-1] for _ in range(self.num_frames - len(rgb_frames))])], axis=0)  # (N, H, W, 3)

        # report texts preprocess
        report_text_list = self.report_text.iloc[idx].split(",")

        np.random.shuffle(report_text_list)
        report_text = "。".join(report_text_list[:35]) + "。"

        # data
        video = self.transforms(list(rgb_frames), return_tensors="pt")["pixel_values"][0]  # (N, 3, H, W)
        text = self.tokenizer([report_text])[0]  # (L,)

        return video, text


def _build_wds_dataset(args, preprocess_fn, is_train, epoch=0, floor=False, tokenizer=None):
    """
    Dataset using webdataset.
    """

    def preprocess_video(transforms, video):
        video_array = mp4_to_ndarray(video, num_frames=args.model.vision.num_frames)  # (N, H, W, 3), unit8
        video_tensor = transforms(list(video_array), return_tensors="pt")["pixel_values"][0]  # (N, 3, H', W')

        return video_tensor

    def preprocess_text(tokenizer, text):
        report_text_list = text["report"]

        np.random.shuffle(report_text_list)

        report_text = "。".join(report_text_list[:35]) + "。"

        return tokenizer([report_text])[0]  # (L,)

    input_shards = args.train_data if is_train else args.val_data

    num_shards = len(wds.shardlists.expand_urls(input_shards))
    assert num_shards >= args.workers * args.world_size, f"number of shards must be >= total workers (= {args.workers}x{args.world_size})"

    num_samples = args.train_num_samples if is_train else args.val_num_samples
    if num_samples is None:
        num_samples = get_dataset_size(input_shards)[0]

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train:
        dataset = wds.WebDataset(
            input_shards,
            shardshuffle=True,
            nodesplitter=wds.split_by_node if args.world_size > 1 else wds.single_node_only,
        )
        dataset = dataset.shuffle(1000)
    else:
        dataset = wds.WebDataset(
            input_shards,
            shardshuffle=False,
            nodesplitter=wds.split_by_node if args.world_size > 1 else wds.single_node_only,
        )

    dataset = dataset.decode()
    dataset = dataset.map_dict(mp4=partial(preprocess_video, preprocess_fn), json=partial(preprocess_text, tokenizer))
    dataset = dataset.to_tuple("mp4", "json")
    dataset = dataset.batched(args.batch_size, partial=not is_train)

    if is_train:
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # info
    dataloader.num_samples = num_samples
    dataloader.num_batches = num_batches

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_size(shards):
    shards_list = wds.shardlists.expand_urls(shards)

    shards_dirs = np.unique([os.path.dirname(path) for path in shards_list])
    total_size = 0
    for dir_path in shards_dirs:
        with open(os.path.join(dir_path, "dataset-size.json"), "r") as f:
            info_dic = json.load(f)
        total_size += info_dic["dataset size"]

    num_shards = len(shards_list)

    return total_size, num_shards


def mp4_to_ndarray(mp4_bytes: bytes, num_frames: int = None) -> np.ndarray:
    """
    MP4のバイナリデータを受け取り、フレームごとにRGB形式のndarrayに変換し、
    [num_frames, height, width, 3] の形状を持つndarrayとして返す。
    """
    if num_frames is not None:
        num_framesx2 = num_frames * 2  # NOTE stride 2

    # mp4_bytes をメモリ上のIOオブジェクトとして開く
    container = av.open(io.BytesIO(mp4_bytes), mode="r")

    frames = []
    for frame in container.decode(video=0):
        rgb_frame = frame.to_rgb()
        arr = rgb_frame.to_ndarray()  # (H, W, 3)
        # arr = cv2.resize(arr, (224, 224))
        frames.append(arr)

        if len(frames) == num_framesx2:
            break

    container.close()

    if num_frames is not None and len(frames) < num_framesx2:
        # 動画が短すぎた場合の処理
        return np.stack(frames + [frames[-1] for _ in range(num_framesx2 - len(frames))], axis=0)[::2]  # (N, H, W, 3) # NOTE stride 2
    return np.stack(frames, axis=0)[::2]  # (N, H, W, 3) # NOTE stride 2
