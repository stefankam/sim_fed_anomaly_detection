import syft as sy
import torch
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_device_data(device_name, data_cache_dir, worker_index=0, num_workers_per_device=1, test_split=0.2):
    """
    Loads the full benign and attack data from the device folder, then splits it into
    train and test sets. The training data is partitioned among workers that share the same
    physical device (i.e. with the same base name).

    Parameters:
      - device_name: A string, e.g. "Danmini_Doorbell_1" or "Ecobee_Thermostat_2"
      - data_cache_dir: The root directory where data is stored.
      - worker_index: Which partition (0-indexed) this worker should receive.
      - num_workers_for_device: Total number of simulated workers for this physical device.
      - test_split: Fraction of data to reserve for testing.


    """
    # Extract the base device name by joining all parts except the last one.
    # For example, "Danmini_Doorbell_1" -> "Danmini_Doorbell"
    parts = device_name.split('_')
    base_device_name = '_'.join(parts[:-1])
    max_benign_samples = 25000
    max_attack_samples = 500

    # Build the folder path using the base device name.
    device_data_dir = os.path.join(data_cache_dir, base_device_name)
    benign_data_path = os.path.join(device_data_dir, "benign_traffic.csv")

    # Load benign data
    benign_data = pd.read_csv(benign_data_path)[:max_benign_samples].fillna(0).to_numpy()
    benign_labels = np.zeros(len(benign_data))  # Label 0 for benign

    # Load attack data (Gafgyt and Mirai)
    attack_data_list = []

    # Gafgyt attacks
    gafgyt_attack_dir = os.path.join(device_data_dir, "gafgyt_attacks")
    if os.path.exists(gafgyt_attack_dir):
        gafgyt_files = [os.path.join(gafgyt_attack_dir, f)
                        for f in os.listdir(gafgyt_attack_dir) if f.endswith(".csv")]
        for file in gafgyt_files:
            df = pd.read_csv(file).fillna(0)
            # Use up to 500 rows from each file
            attack_data_list.append(df.iloc[:max_attack_samples].to_numpy())  # Adjusting sample size

    # Mirai attacks
    mirai_attack_dir = os.path.join(device_data_dir, "mirai_attacks")
    if os.path.exists(mirai_attack_dir):
        mirai_files = [os.path.join(mirai_attack_dir, f)
                       for f in os.listdir(mirai_attack_dir) if f.endswith(".csv")]
        for file in mirai_files:
            df = pd.read_csv(file).fillna(0)
            # Use up to 500 rows from each file
            attack_data_list.append(df.iloc[:max_attack_samples].to_numpy())  # Adjusting sample size

    # Combine all attack data if available
    if attack_data_list:
        attack_data = np.vstack(attack_data_list)
        attack_labels = np.ones(len(attack_data))  # Label 1 for attacks
    else:
        attack_data = np.empty((0, benign_data.shape[1]))
        attack_labels = np.empty(0)

    # Combine benign and attack data
    combined_data = np.vstack([benign_data, attack_data])
    combined_labels = np.concatenate([benign_labels, attack_labels])

    # Split the data into training and test sets (using stratification to maintain balance)
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, combined_labels, test_size=test_split, stratify=combined_labels, random_state=42
    )
    print(f"Total X_train: {len(X_train)}, num_workers: {num_workers_per_device}")
    print(f"Total X_test: {len(X_test)}, num_workers: {num_workers_per_device}")

    # Partition the training data among the simulated workers for this device.
    # Partition the dataset among workers
    X_train_partitions = np.array_split(X_train, num_workers_per_device)
    y_train_partitions = np.array_split(y_train, num_workers_per_device)

    X_test_partitions = np.array_split(X_test, num_workers_per_device)  # Now splitting test set
    y_test_partitions = np.array_split(y_test, num_workers_per_device)

    # Assign only the subset for this worker
    X_train_worker = X_train_partitions[worker_index]
    y_train_worker = y_train_partitions[worker_index]

    X_test_worker = X_test_partitions[worker_index]  # Each worker gets a part of test data
    y_test_worker = y_test_partitions[worker_index]


    # Convert to torch tensors
    features_train = torch.tensor(X_train_worker, dtype=torch.float32)
    labels_train = torch.tensor(y_train_worker, dtype=torch.float32)
    features_test = torch.tensor(X_test_worker, dtype=torch.float32)
    labels_test = torch.tensor(y_test_worker, dtype=torch.float32)

    return features_train, labels_train, features_test, labels_test





def load_device_data_extra(device_name, data_cache_dir, worker_index=0, num_workers_per_device=1, test_split=0.2,
                     attack_weight_factor=2):
    """
    Loads the full benign and attack data from the device folder, then splits it into
    train and test sets. More attack data can be assigned to specific workers.

    Parameters:
      - device_name: A string, e.g. "Danmini_Doorbell_1" or "Ecobee_Thermostat_2"
      - data_cache_dir: The root directory where data is stored.
      - worker_index: Which partition (0-indexed) this worker should receive.
      - num_workers_per_device: Total number of simulated workers for this physical device.
      - test_split: Fraction of data to reserve for testing.
      - attack_weight_factor: Controls how much attack data certain workers receive.

    Returns:
      - features_train, labels_train, features_test, labels_test (as torch tensors)
    """

    # Extract the base device name
    parts = device_name.split('_')
    base_device_name = '_'.join(parts[:-1])

    # Build paths
    device_data_dir = os.path.join(data_cache_dir, base_device_name)
    benign_data_path = os.path.join(device_data_dir, "benign_traffic.csv")

    # Load benign data
    benign_data = pd.read_csv(benign_data_path)[:25000].fillna(0).to_numpy()
    benign_labels = np.zeros(len(benign_data))  # Label 0 for benign

    # Load attack data
    attack_data_list = []

    # Gafgyt attacks
    gafgyt_attack_dir = os.path.join(device_data_dir, "gafgyt_attacks")
    if os.path.exists(gafgyt_attack_dir):
        for file in os.listdir(gafgyt_attack_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(gafgyt_attack_dir, file)).fillna(0)
                attack_data_list.append(df.iloc[:2500].to_numpy())

    # Mirai attacks
    mirai_attack_dir = os.path.join(device_data_dir, "mirai_attacks")
    if os.path.exists(mirai_attack_dir):
        for file in os.listdir(mirai_attack_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(mirai_attack_dir, file)).fillna(0)
                attack_data_list.append(df.iloc[:2500].to_numpy())

    # Combine attack data
    if attack_data_list:
        attack_data = np.vstack(attack_data_list)
        attack_labels = np.ones(len(attack_data))  # Label 1 for attacks
    else:
        attack_data = np.empty((0, benign_data.shape[1]))
        attack_labels = np.empty(0)

    # Combine benign and attack data
    combined_data = np.vstack([benign_data, attack_data])
    combined_labels = np.concatenate([benign_labels, attack_labels])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, combined_labels, test_size=test_split, stratify=combined_labels, random_state=42
    )

    # Partition training data for workers
    benign_train = X_train[y_train == 0]
    attack_train = X_train[y_train == 1]

    num_benign = len(benign_train)
    num_attack = len(attack_train)

    benign_splits = np.array_split(benign_train, num_workers_per_device)
    attack_splits = np.array_split(attack_train, num_workers_per_device)

    # Assign more attack data to specific workers
    X_train_worker = benign_splits[worker_index]
    y_train_worker = np.zeros(len(X_train_worker))

    # Increase attack data for certain workers
    if worker_index < min(attack_weight_factor, num_workers_per_device):
        extra_attack_data = attack_splits[worker_index]
    else:
        extra_attack_data = attack_splits[-1]

    X_train_worker = np.vstack([X_train_worker, extra_attack_data])
    y_train_worker = np.concatenate([y_train_worker, np.ones(len(extra_attack_data))])

    # Partition test data evenly
    X_test_splits = np.array_split(X_test, num_workers_per_device)
    y_test_splits = np.array_split(y_test, num_workers_per_device)

    X_test_worker = X_test_splits[worker_index]
    y_test_worker = y_test_splits[worker_index]

    # Convert to tensors
    features_train = torch.tensor(X_train_worker, dtype=torch.float32)
    labels_train = torch.tensor(y_train_worker, dtype=torch.float32)
    features_test = torch.tensor(X_test_worker, dtype=torch.float32)
    labels_test = torch.tensor(y_test_worker, dtype=torch.float32)

    print(
        f"Worker {worker_index}: Train Size = {len(features_train)}, Attack % = {sum(y_train_worker) / len(y_train_worker):.2%}")

    return features_train, labels_train, features_test, labels_test



# Simulate 4 workers, with the first 2 receiving extra attack data
#for i in range(5):
#    train_x, train_y, test_x, test_y = load_device_data_extra(
#        "Danmini_Doorbell_1", "/Users/stefanbehfar/Documents/Projects/FedML/iot/anomaly_detection_for_cybersecurity/data_cache", worker_index=i, num_workers_per_device=5, attack_weight_factor=2
#    )