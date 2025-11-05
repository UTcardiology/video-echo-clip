import os
import random
import re
import numpy as np
import torch
import subprocess
import glob
import fsspec
import logging


def random_seed(seed=7, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]  # "epoch_1.pt" -> ["epoch_", 1, ".pt"]


def get_latest_checkpoint(path: str, epoch: int = None):
    checkpoints = glob.glob(path + "**/*.pt", recursive=True)

    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)  # sort checkpoints by newest first ([epoch_1.pt, epoch_2.pt, ...])
        if epoch is None:
            return checkpoints[-1]
        else:
            for checkpoint in checkpoints:
                if f"epoch_{epoch}.pt" in checkpoint:
                    return checkpoint
            return None
    return None


def pt_load(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def setup_logging(log_file, level):
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S")

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
