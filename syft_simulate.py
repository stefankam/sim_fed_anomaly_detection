import syft as sy
import torch
import pandas as pd
import os
import numpy as np


# Define Workers
device_names = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat"
]
devices = [sy.Worker(name=name) for name in device_names]
server = sy.Worker(name="server")



# Load and Split Data for Each Device
def load_device_data(device_name, data_cache_dir):
    # ReDefine Dataset Directory
    data_cache_dir = "/Users/stefanbehfar/Documents/Projects/FedML/iot/anomaly_detection_for_cybersecurity/data_cache/"
    device_data_dir = os.path.join(data_cache_dir, device_name)
    benign_data_path = os.path.join(device_data_dir, "benign_traffic.csv")

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

