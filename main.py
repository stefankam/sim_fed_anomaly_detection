import torch
import syft as sy
import sys
import argparse
import numpy as np
sys.path.append("/Users/stefanbehfar/Documents/Projects/FedML/iot/anomaly_detection_for_cybersecurity/")
from sklearn.preprocessing import StandardScaler

from utils import plot_reconstruction_loss
from utils import evaluate_local_and_global
from utils import evaluate_anomalies
from utils import define_threshold, get_threshold_global
from utils import generate_report
from utils import plot_error_distributions
from utils import plot_precision_recall

from syft_simulate import load_device_data
from model.autoencoder import AutoEncoder
from torch import nn
from sklearn.model_selection import train_test_split
from data.data_loader import prepare_datasets

# Create Workers
device_names = ["Danmini_Doorbell", "Ecobee_Thermostat"]
workers = [sy.Worker(name=name) for name in device_names]
server = sy.Worker(name="server")

parser = argparse.ArgumentParser(description="Federated Learning Arguments")

parser.add_argument("--data_cache_dir", type=str, default="./data", help="Path to data cache directory")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")

args, unknown = parser.parse_known_args()  # Ignores PyCharm-specific args

# Prepare Datasets (only run once)
data_cache_dir = "./data_cache"
#prepare_datasets(args, device_names, workers)

# Load Data for Each Device
for worker in workers:
    #data_key = f"{worker.name}_dataset"
    features, labels = load_device_data(worker.name, data_cache_dir)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)
    # Store data in worker's client cache
    data = {"train": {"features": X_train, "labels": y_train}, "test": {"features": X_test, "labels": y_test}}
    worker.client_cache.update(data_key = data)
    # Fit a global scaler once on all training data
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
num_rounds = 3
output_dim = 115
epochs_per_round = 5
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
precision, recall, f1 = evaluate_anomalies(global_autoencoder, global_scaler, workers, threshold)
plot_error_distributions(global_autoencoder, global_scaler, workers, threshold)
plot_precision_recall(global_autoencoder, global_scaler, workers)

