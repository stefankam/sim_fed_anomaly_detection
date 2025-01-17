import syft as sy
import torch
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Define Workers
device_names = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat"
]
devices = [sy.Worker(name=name) for name in device_names]
server = sy.Worker(name="server")

# Step 2: Define Dataset Directory
data_cache_dir = "./data_cache"


# Step 3: Load and Split Data for Each Device
def load_device_data(device_name, data_cache_dir):
    device_data_dir = os.path.join(data_cache_dir, device_name)
    benign_data_path = os.path.join(device_data_dir, "benign_traffic.csv")

    if not os.path.exists(benign_data_path):
        raise FileNotFoundError(f"Dataset for {device_name} not found at {benign_data_path}")

    # Load benign data
    benign_data = pd.read_csv(benign_data_path).fillna(0).to_numpy()
    benign_labels = torch.zeros(len(benign_data))  # All benign labels = 0

    # Simulate attack data
    #attack_data_dir = os.path.join(device_data_dir, "gafgyt_attacks")
    #attack_files = [os.path.join(attack_data_dir, f) for f in os.listdir(attack_data_dir) if f.endswith(".csv")]
    #attack_data = pd.concat([pd.read_csv(f).fillna(0).iloc[:500] for f in attack_files]).to_numpy()
    #attack_labels = torch.ones(len(attack_data))  # All attack labels = 1

    # Load attack data (Gafgyt + Mirai)
    attack_data_list = []

    # Gafgyt attacks
    gafgyt_attack_dir = os.path.join(device_data_dir, "gafgyt_attacks")
    if os.path.exists(gafgyt_attack_dir):
        gafgyt_files = [os.path.join(gafgyt_attack_dir, f) for f in os.listdir(gafgyt_attack_dir) if f.endswith(".csv")]
        for file in gafgyt_files:
            attack_data_list.append(pd.read_csv(file).fillna(0).iloc[:500].to_numpy())

    # Mirai attacks
    mirai_attack_dir = os.path.join(device_data_dir, "mirai_attacks")
    if os.path.exists(mirai_attack_dir):
        mirai_files = [os.path.join(mirai_attack_dir, f) for f in os.listdir(mirai_attack_dir) if f.endswith(".csv")]
        for file in mirai_files:
            attack_data_list.append(pd.read_csv(file).fillna(0).iloc[:500].to_numpy())

    # Combine all attack data
    if attack_data_list:
        attack_data = np.vstack(attack_data_list)
        attack_labels = torch.ones(len(attack_data))  # All attack labels = 1
    else:
        attack_data = np.empty((0, benign_data.shape[1]))
        attack_labels = torch.empty(0)

    # Combine benign and attack data
    features = torch.tensor(np.vstack([benign_data, attack_data]), dtype=torch.float32)
    labels = torch.cat([torch.tensor(benign_labels), torch.tensor(attack_labels)])

    return features, labels


# Step 4: Distribute Data to Workers
for device, device_name in zip(devices, device_names):
    features, labels = load_device_data(device_name, data_cache_dir)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)

    # Store data in worker's client cache
    data = {"train": {"features": X_train, "labels": y_train}, "test": {"features": X_test, "labels": y_test}}
    device.client_cache.update(value=data, description=f"Local dataset for {device.name}")


# Step 5: Simulate Local Training
def train_local_model(worker, data_key):
    # Retrieve local training data
    data = worker.client_cache.get(data_key)
    train_data = data["train"]
    features, labels = train_data["features"], train_data["labels"]

    # Simulate local training: Compute the mean of features as the local update
    local_update = features.mean(dim=0)  # Example: Simple average
    worker.client_cache.update(value=local_update, description=f"Model update for {worker.name}")
    return local_update


# Perform local training on each device
updates = []
for device in devices:
    local_update = train_local_model(device, "value")
    updates.append(local_update)


# Step 6: Aggregate Updates on Server
def aggregate_updates(server, updates):
    global_update = torch.stack(updates).mean(dim=0)  # Federated averaging
    server.client_cache.update(value=global_update, description="Global model update")
    return global_update


# Aggregate updates
global_model = aggregate_updates(server, updates)
print(f"Global model parameters: {global_model}")


# Step 7: Distribute Global Model to Devices
def distribute_global_model(devices, server, global_key):
    global_model = server.client_cache.get(global_key)
    for device in devices:
        device.client_cache.update(value=global_model, description="Updated global model")


# Distribute model
distribute_global_model(devices, server, "value")

# Example: Verify model update on a device
for device in devices:
    updated_model = device.client_cache.get("value")
    print(f"Updated model on {device.name}: {updated_model}")
