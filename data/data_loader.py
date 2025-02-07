import os
import urllib.request
import zipfile
import logging

import numpy as np
import pandas as pd
import torch
import syft as sy


def download_data(data_cache_dir, device_name):
    url_root = "https://fediot.s3.us-west-1.amazonaws.com/fediot"
    url = os.path.join(url_root, (device_name + ".zip"))
    saved_path = os.path.join(data_cache_dir, (device_name + ".zip"))
    urllib.request.urlretrieve(url, saved_path)
    with zipfile.ZipFile(saved_path, "r") as f:
        f.extractall(data_cache_dir)


def prepare_datasets(args, device_list, workers):

    benign_data_local_dict = {}
    attack_data_local_dict = {}
    benign_data_local_num_dict = {}
    benign_data_num = 0
    attack_data_num = 0

    # Normalize dataset using pre-calculated min/max
    min_max_file_path = "/Users/stefanbehfar/Documents/Projects/FedML/iot/anomaly_detection_for_cybersecurity/data"
    min_dataset = np.loadtxt(os.path.join(min_max_file_path, "min_dataset.txt"))
    max_dataset = np.loadtxt(os.path.join(min_max_file_path, "max_dataset.txt"))

    # Ensure data cache directory exists
    if not os.path.exists(args.data_cache_dir):
        os.makedirs(args.data_cache_dir, exist_ok=True)

    # Iterate over devices and create datasets
    for i, (device_name, worker) in enumerate(zip(device_list, workers)):
        device_data_cache_dir = os.path.join(args.data_cache_dir, device_name)
        if not os.path.exists(device_data_cache_dir):
            os.makedirs(device_data_cache_dir, exist_ok=True)
            logging.info(f"Downloading dataset for device: {device_name}")
            download_data(args.data_cache_dir, device_name)

        logging.info(f"Creating dataset for: {device_name}")

        # Load benign traffic data
        benign_data = pd.read_csv(
            os.path.join(device_data_cache_dir, "benign_traffic.csv")
        )
        benign_data = benign_data.fillna(0).to_numpy()
        benign_data = (benign_data - min_dataset) / (max_dataset - min_dataset)

        # Load attack data
        attack_data_dir = os.path.join(device_data_cache_dir, "gafgyt_attacks")
        attack_data_files = [
            os.path.join(attack_data_dir, f)
            for f in os.listdir(attack_data_dir)
            if f.endswith(".csv")
        ]

        if device_name not in ["Ennio_Doorbell", "Samsung_SNH_1011_N_Webcam"]:
            mirai_attack_dir = os.path.join(device_data_cache_dir, "mirai_attacks")
            mirai_attack_files = [
                os.path.join(mirai_attack_dir, f)
                for f in os.listdir(mirai_attack_dir)
                if f.endswith(".csv")
            ]
            attack_data_files.extend(mirai_attack_files)

        # Load attack data
        attack_data = pd.concat(
            [pd.read_csv(f).fillna(0) for f in attack_data_files]
        )
        attack_data = (attack_data - attack_data.mean()) / (attack_data.std())
        attack_data = attack_data.to_numpy()

        # Split attack data among workers
        attack_splits = np.array_split(attack_data, len(workers))
        attack_data = attack_splits[i]  # Each worker gets a unique subset

        # Create DataLoader objects
        benign_loader = torch.utils.data.DataLoader(
            benign_data, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        attack_loader = torch.utils.data.DataLoader(
            attack_data, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        # Store datasets in worker's client cache
        worker.client_cache.update(
            value={"benign": benign_loader, "attack": attack_loader},
            description=f"Local dataset for {device_name}",
        )

        # Update statistics
        benign_data_local_dict[i] = benign_loader
        attack_data_local_dict[i] = attack_loader
        benign_data_local_num_dict[i] = len(benign_loader)
        benign_data_num += len(benign_loader)
        attack_data_num += len(attack_loader)

    # Return dataset statistics for analysis
    dataset = {
        "benign_data_num": benign_data_num,
        "attack_data_num": attack_data_num,
        "benign_data_local_dict": benign_data_local_dict,
        "attack_data_local_dict": attack_data_local_dict,
        "benign_data_local_num_dict": benign_data_local_num_dict,
        "class_num": 115,
    }
    return dataset