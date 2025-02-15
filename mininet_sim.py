import torch
import sys
import argparse
import numpy as np
import subprocess
import re
sys.path.append("/home/sim_fed_anomaly_detection")
from sklearn.preprocessing import StandardScaler

from utils import plot_reconstruction_loss
from utils import evaluate_local_and_global
from utils import evaluate_anomalies
from utils import define_threshold, get_threshold_global
from utils import generate_report
from utils import plot_error_distributions
from utils import plot_precision_recall
from utils import tune_threshold

from syft_simulate import load_device_data
from model.autoencoder import AutoEncoder
from torch import nn
from sklearn.model_selection import train_test_split
from data.data_loader import prepare_datasets

from mininet.cli import CLI
from mininet.net import Mininet
from mininet.node import Host,  OVSController, OVSSwitch
from mininet.link import TCLink
import math
import psutil
import time
import matplotlib.pyplot as plt
import threading
from tabulate import tabulate

class CustomHost(Host):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        # Custom attribute for client cache
        self.client_cache = {}

class TopologyProvider:
    def __init__(self, device_names, num_workers, workers, link_latency=None, link_loss=None):
        self.device_names = device_names
        self.num_workers = num_workers
        self.link_latency = link_latency
        self.link_loss = link_loss
        self.workers = workers
        self.latency_data = []
        self.packet_loss_data = []
        self.throughput_data = []
        self.resource_data = []
        self.timestamps = []
        self.running = True

        # Handle latency: divide by 2 since it's a star topology
        self.link_latency = f"{link_latency / 2}ms" if link_latency else None

        # Handle loss: 1 - sqrt(1 - L) to adjust for double-link effect
        self.link_loss = (1 - math.sqrt(1 - link_loss / 100)) * 100 if link_loss else None
        if self.link_loss == 0:
            self.link_loss = None

        self.switch_num = 0
        self.host_num = 0

    def add_switch(self):
        name = f"s{self.switch_num + 1}"
        self.switch_num += 1
        return self.net.addSwitch(name)

    def add_worker(self, device, index, *args, **kwargs):
        if not hasattr(self, "net") or self.net is None:
            raise ValueError("Mininet instance (self.net) is not initialized. Call setup() first.")
        worker_name = f"{device}_{index + 1}"
        self.host_num += 1
        # Assign a unique IP address dynamically
        ip_address = f"10.0.0.{self.host_num}"  # Start from 10.0.0.1, 10.0.0.2, etc.
        # Create the host with the given IP address and other parameters
        host = self.net.addHost(worker_name, cls=CustomHost, ip=ip_address, **kwargs)

        return host

    def monitor_latency_packet_loss(self, host):
        """Ping a host to measure latency and packet loss."""
        try:
            output = host.cmd(f"ping -c 5 {host.IP()}")
            print("host.name: ", host.name)
            print("host.IP: ", host.IP())
            print("ping output: ", output)
            # Extract packet loss
            loss_match = re.search(r'(\d+)% packet loss', output)
            loss = float(loss_match.group(1)) if loss_match else None
            # Extract latency (min/avg/max/mdev)
            latency_match = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', output)
            latency = float(latency_match.group(2)) if latency_match else None  # Extract avg latency
            if loss is not None:
               self.packet_loss_data.append(loss)
            if latency is not None:
               self.latency_data.append(latency)

            print(f"Latency: {latency} ms, Packet Loss: {loss}%")
        
        except Exception as e:
            print(f"Error monitoring latency: {e}")

    def start_iperf(self, host):
        """Measure network throughput using iperf."""
        # Start iperf server on the host using host.cmd()
        print(f"Starting iperf server on {host.name}")
        host.cmd("iperf -s &")
        time.sleep(2)  # Give time for the server to start

        # Run the iperf client on another host
        print(f"Running iperf client from {host.name}")
        result = host.cmd(f"iperf -c {host.IP()} -t 5")

        # Print the raw output for debugging
        print(f"Raw iperf output:\n{result}")
        
        # Parse the output to get throughput
        # Check for different units in the output
        for line in result.splitlines():
            if "Mbits/sec" in line:
                throughput = float(line.split("Mbits/sec")[0].split()[-1])
                throughput_unit = "Mbits/sec"
                break
            elif "Gbits/sec" in line:
                throughput = float(line.split("Gbits/sec")[0].split()[-1]) * 1000  # Convert Gbits/sec to Mbits/sec
                throughput_unit = "Gbits/sec"
                break
        if throughput is not None:
            self.throughput_data.append(throughput)
            print(f"Throughput: {throughput} Mbits/sec (from {throughput_unit})")
        else:
            print("Error: Could not parse iperf output.")
    
    def monitor_resources(self):
        """Monitor system resource usage."""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        self.resource_data.append((cpu, memory))
        self.timestamps.append(time.time())
        print(f"CPU Usage: {cpu}%")
        print(f"Memory Usage: {memory.percent}%")
    
    def generate_report(self):
        """Generate a report with latency, throughput, and resource usage graphs."""
        plt.figure(figsize=(10, 5))
        plt.subplot(3, 1, 1)
        plt.plot(self.latency_data, label="Latency (ms)")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.throughput_data, label="Throughput (Mbits/sec)")
        plt.legend()
        plt.subplot(3, 1, 3)
        cpu_usage = [x[0] for x in self.resource_data]
        memory_usage = [x[1] for x in self.resource_data]
        plt.plot(cpu_usage, label="CPU Usage (%)")
        plt.plot(memory_usage, label="Memory Usage (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def generate_table(self):
        # Extract CPU and Memory usage data
        cpu_usage = [cpu for cpu, _ in self.resource_data]
        memory_usage = [mem.percent for _, mem in self.resource_data]
        
        def safe_stats(data):
            """Returns min, max, and average safely even if the data is empty."""
            if not data:
                return "N/A", "N/A", "N/A"
            return min(data), max(data), sum(data)/len(data)

        # Prepare report data with safe checks
        report_data = [
        ["Metric", "Min", "Max", "Average"],
        ["Latency (ms)", *safe_stats(self.latency_data)],
        ["Packet Loss (%)", *safe_stats(self.packet_loss_data)],
        ["Throughput (Mbps)", *safe_stats(self.throughput_data)],
        ["CPU Usage (%)", *safe_stats(cpu_usage)],
        ["Memory Usage (%)", *safe_stats(memory_usage)],
        ]

        print(tabulate(report_data, headers="firstrow", tablefmt="grid"))

    def setup(self):
        print("🚀 Starting Mininet Setup...")

        # Initialize Mininet with default OVSController
        self.net = Mininet(controller=OVSController, switch=OVSSwitch, link=TCLink)
        self.net.addController("c0")  # Use the default controller

        # Create a single switch
        switch = self.add_switch()
        print(f"✅ Switch created: {switch}")

        # Add workers for each device
        for device in self.device_names:
            for i in range(self.num_workers):
                worker = self.add_worker(device, i)
                self.workers.append(worker)
                print(f"🔗 Linked {worker} to switch {switch}")
                self.net.addLink(worker, switch, delay=self.link_latency, loss=self.link_loss)

        # Start the network
        self.net.start()
        print("✅ Mininet started successfully!")

        # After the network is started, start iperf tests for each worker
        for worker in self.workers:
           self.start_iperf(worker)
           self.monitor_resources()
           self.monitor_latency_packet_loss(worker)

        return self.net



# Define devices and number of workers
device_names = ["Doorbell", "Thermost"]
num_workers = 1  
link_latency = 2
link_loss = 0.01
workers = []
topology = TopologyProvider(device_names, num_workers, workers, link_latency, link_loss)

# Setup and start Mininet
net = topology.setup()

server = "server"

parser = argparse.ArgumentParser(description="Federated Learning Arguments")
parser.add_argument("--data_cache_dir", type=str, help="Path to data cache directory")
parser.add_argument("--batch_size", type=int, help="Batch size for training")

args, unknown = parser.parse_known_args()  # Ignores PyCharm-specific args

# Prepare Datasets (only run once)
args.data_cache_dir = "/home/sim_fed_anomaly_detection/data_cache"
args.batch_size = 2
#prepare_datasets(args, device_names, workers)

# Container to hold training features from all workers
all_train_features = []

# Load Data for Each Device
for worker in workers:
    device_name, worker_num = worker.name.rsplit("_", 1)  # Extract device name & worker number
    worker_index = int(worker_num) - 1  # Convert 1-based to 0-based (0-4 per device)

    X_train, y_train, X_test, y_test = load_device_data(
        worker.name, args.data_cache_dir,
        worker_index=worker_index,  # Correct local worker index
        num_workers_per_device=num_workers,
        test_split=0.2
    )

    # Store data in worker's client cache
    data = {"train": {"features": X_train, "labels": y_train}, "test": {"features": X_test, "labels": y_test}}
    worker.client_cache.update(data_key=data)

    print(f"[{worker.name}] X_train {len(X_train)}, X_test: {len(X_test)}, Y_train {len(y_train)}, y_test {len(y_test)}")

    benign_train = (y_train == 0).sum().item()
    attack_train = (y_train == 1).sum().item()
    benign_test = (y_test == 0).sum().item()
    attack_test = (y_test == 1).sum().item()
    total_train = len(y_train)
    total_test = len(y_test)

    print(f"{worker.name} - TRAIN: Total {total_train} (Benign: {benign_train}, Attack: {attack_train})")
    print(f"{worker.name} - TEST: Total {total_test} (Benign: {benign_test}, Attack: {attack_test})")


all_features = np.vstack([worker.client_cache.get('data_key')['train']['features'] for worker in workers])
global_scaler = StandardScaler().fit(all_features)



# Train AutoEncoders
def train_local_autoencoder(worker_model, worker, output_dim=115, epochs=5, lr=0.001, batch_size=32):
    # Retrieve data from the client cache
    data = worker.client_cache.get('data_key')

    features = data['train']['features']
    #features = (features - features.min()) / (features.max() - features.min())
    # **Use global scaler**
    features = global_scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    # Create a DataLoader for mini-batching
    dataset = torch.utils.data.TensorDataset(features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize AutoEncoder and optimizer
    autoencoder = AutoEncoder(output_dim=output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

  
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch_features = batch[0]  # Extract features from DataLoader batch
            autoencoder.train()
            optimizer.zero_grad()

            # Forward pass and compute reconstruction loss
            reconstructed = worker_model(batch_features)
            loss = criterion(reconstructed, batch_features)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        # Call monitor_resources periodically (e.g., every few epochs)
        if epoch % 5 == 0:  # Call every 5 epochs
            topology.monitor_resources()
        print(f"[{worker.name}] Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save the trained model parameters
    worker.client_cache.update(
        local_value=autoencoder.state_dict(),
        description=f"Trained AutoEncoder for {worker.name}",
    )
    return autoencoder.state_dict(), epoch_loss/len(dataloader)


# Aggregate Updates to perform weighted averaging based on the number of samples from each worker
def aggregate_updates(local_updates, global_model, local_sample_counts):
    state_dict = global_model.state_dict()
    total_samples = sum(local_sample_counts)

    for key in state_dict.keys():
        weighted_sum = sum(local_updates[i][key] * (local_sample_counts[i] / total_samples) for i in range(len(local_updates)))
        state_dict[key] = weighted_sum

    return state_dict


# Evaluation
def evaluate_global_model(global_model, workers):
    total_loss = 0
    criterion = nn.MSELoss()

    for worker in workers:
        data = worker.client_cache.get('data_key')
        if data is None:
            raise ValueError(f"No data found for {worker.name}")
        features = data["test"]["features"]  # Or use a separate test key if available
        #features = (features - features.min()) / (features.max() - features.min())
        # **Use global scaler**
        features = global_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(features)
            loss = criterion(reconstructed, features)
            total_loss += loss.item()

    avg_loss = total_loss / len(workers)
    print(f"Average Test Loss: {avg_loss:.4f}")
    return avg_loss


# Implementation
num_rounds = 2
output_dim = 115
epochs_per_round = 10
lr = 0.001

# Step 1: Initialize the global model
global_autoencoder = AutoEncoder(output_dim=output_dim)

global_losses = []
local_losses_per_round = []
for round_num in range(1, num_rounds + 1):
    print(f"\n=== Federated Round {round_num} ===")

    # Step 1: Distribute the global model to devices
    global_state = global_autoencoder.state_dict()
    for worker in workers:
        worker.client_cache.update(
            global_value=global_state,
            description="Global AutoEncoder Model"
        )

    # Step 2: Local Training on Devices
    local_losses = []
    local_updates = []
    local_sample_counts = []  # Store worker dataset sizes

    for worker in workers:
        # Retrieve the latest global model
        global_state = worker.client_cache.get("global_value")
        if global_state is None:
            raise ValueError(f"Global model not found in {worker.name}")
        worker_model = AutoEncoder(output_dim=output_dim)
        worker_model.load_state_dict(global_state)

        # Train on worker-specific data
        worker_state_before = worker_model.state_dict()  # Save initial state

        # **Train the local model and get loss**
        _,local_loss = train_local_autoencoder(worker_model, worker, output_dim=output_dim, epochs=epochs_per_round, lr=lr)
        local_losses.append(local_loss)  # Store the local loss

        worker_state_after = worker_model.state_dict()  # Get trained state

        # Compute model update (delta)
        #model_update = {k: worker_state_after[k] - worker_state_before[k] for k in worker_state_before}
        model_update = {k: worker_state_after[k].detach().clone() - worker_state_before[k].detach().clone()
                        for k in worker_state_before}
        local_updates.append(model_update)
        data = worker.client_cache.get("data_key")
        local_sample_counts.append(len(data["train"]["features"]))  # Store dataset size

    # Step 3: Aggregate Updates using Federated Averaging
    global_state = aggregate_updates(local_updates, global_autoencoder, local_sample_counts)
    global_autoencoder.load_state_dict(global_state)

    # Compute global loss (optional: use a test dataset)
    global_loss = evaluate_global_model(global_autoencoder,workers)

    # Store losses for plotting
    global_losses.append(global_loss)

    # Compute and print average local loss
    if local_losses:  # Ensure it's not empty
        avg_local_loss = sum(local_losses) / len(local_losses)
    else:
        avg_local_loss = 0.0  # Default value

    print(f"Round {round_num}: Global Loss = {global_loss:.4f}, Avg Local Loss = {avg_local_loss:.4f}")


evaluate_global_model(global_autoencoder, workers)

# Evaluate and Visualize Results
print("\nEvaluating the global model...")
global_avg_loss = evaluate_global_model(global_autoencoder, workers)

print("\nPlotting reconstruction loss distribution...")
loss_data = plot_reconstruction_loss(global_autoencoder, global_scaler, workers)

print("\nComparing local and global model performance...")
local_global_comparison = evaluate_local_and_global(global_autoencoder, global_scaler, workers)

print("\nGenerating final report...")
generate_report(global_avg_loss, local_global_comparison)


threshold = define_threshold(global_autoencoder, global_scaler, workers)
#threshold = get_threshold_global(global_autoencoder, args, device_names)
accuracy, precision, recall, f1 = evaluate_anomalies(global_autoencoder, global_scaler, workers, threshold)
plot_error_distributions(global_autoencoder, global_scaler, workers, threshold)
plot_precision_recall(global_autoencoder, global_scaler, workers)

best_threshold, best_metrics = tune_threshold(global_autoencoder, global_scaler, workers, num_thresholds=100)
plot_error_distributions(global_autoencoder, global_scaler, workers, best_threshold, data_key="data_key")

topology.generate_report()
topology.generate_table()

def count_samples(loader):
    # Try to use the dataset attribute
    if hasattr(loader, "dataset"):
        try:
            return len(loader.dataset)
        except Exception:
            pass
    # If .dataset is not available or fails, iterate over the loader
    total = 0
    for batch in loader:
        # If the loader returns a tuple (e.g., (features, labels))
        if isinstance(batch, (tuple, list)):
            # Assume batch[0] contains the features and its first dimension is the batch size
            total += batch[0].shape[0]
        else:
            # Otherwise assume the batch itself is a tensor or list-like
            total += len(batch)
    return total

# Example usage:
for worker in workers:
    data = worker.client_cache.get('data_key')
    if data is None:
        print(f"{worker.name} has no data in the cache.")
        continue

    # Retrieve DataLoaders from the cache
    train_loader = data.get("train")
    test_loader = data.get("test")

    benign_count = count_samples(train_loader)
    attack_count = count_samples(test_loader)
    total = benign_count + attack_count

    print(f"{worker.name} - Total Samples: {total}, Benign: {benign_count}, Attack: {attack_count}")
