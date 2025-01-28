import torch
import syft as sy
import sys
sys.path.append("/Users/stefanbehfar/Documents/Projects/FedML/iot/anomaly_detection_for_cybersecurity")

from utils import plot_reconstruction_loss
from utils import evaluate_local_and_global
from utils import evaluate_anomalies
from utils import define_threshold
from utils import generate_report
from utils import plot_error_distributions
from utils import plot_precision_recall

from syft_simulate import load_device_data
from model.autoencoder import AutoEncoder
from torch import nn
from sklearn.model_selection import train_test_split

# Step 1: Create Workers
device_names = ["Danmini_Doorbell", "Ecobee_Thermostat"]
workers = [sy.Worker(name=name) for name in device_names]
server = sy.Worker(name="server")

# Step 2: Prepare Datasets (only run once)
data_cache_dir = "./data_cache"
#prepare_datasets(data_cache_dir)

# Step 3: Load Data for Each Device
for worker in workers:
    #data_key = f"{worker.name}_dataset"
    features, labels = load_device_data(worker.name, data_cache_dir)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)
    # Store data in worker's client cache
    data = {"train": {"features": X_train, "labels": y_train}, "test": {"features": X_test, "labels": y_test}}
    worker.client_cache.update(data_key = data)

# Step 4: Train AutoEncoders
def train_local_autoencoder(worker, data_key='value', output_dim=115, epochs=5, lr=0.001, batch_size=32):
    # Retrieve data from the client cache
    print("worker:", worker)
    print("data_key:", data_key)
    data = worker.client_cache.get('data_key')
    assert data is not None, "Data is None. Check the `data_key` and cache contents."
    assert "train" in data, "Key 'train' is missing in the data."
    assert "features" in data["train"], "Key 'features' is missing in the train data."

    features = data['train']['features']
    features = (features - features.min()) / (features.max() - features.min())


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
            reconstructed = autoencoder(batch_features)
            loss = criterion(reconstructed, batch_features)
            #print("epoch: ", epoch)
            #print("loss: ", loss)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[{worker.name}] Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save the trained model parameters
    worker.client_cache.update(
        value=autoencoder.state_dict(),
        description=f"Trained AutoEncoder for {worker.name}",
    )
    return autoencoder.state_dict()


# Step 5: Aggregate Updates
def aggregate_updates(updates, global_model):
    state_dict = global_model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = torch.stack([update[key] for update in updates]).mean(dim=0)
    return state_dict

num_rounds = 1
output_dim = 115
epochs_per_round = 1
lr = 0.001

# Initialize the global model
global_autoencoder = AutoEncoder(output_dim=output_dim)

for round_num in range(1, num_rounds + 1):
    print(f"\n=== Federated Round {round_num} ===")

    # Step 1: Distribute the global model to devices
    global_state = global_autoencoder.state_dict()
    for worker in workers:
        worker.client_cache.update(value=global_state, description="Global AutoEncoder Model")

    # Step 2: Local Training on Devices
    local_updates = []
    for worker in workers:
        local_update = train_local_autoencoder(worker, data_key='value', output_dim=output_dim, epochs=epochs_per_round, lr=lr)
        local_updates.append(local_update)

    # Step 3: Aggregate Updates
    global_state = aggregate_updates(local_updates, global_autoencoder)
    global_autoencoder.load_state_dict(global_state)

    print(f"Round {round_num}: Global model updated and distributed.")

print("\nFederated learning completed.")


# Evaluate the globa model
def evaluate_global_model(global_model, workers, data_key="value"):
    total_loss = 0
    criterion = nn.MSELoss()

    for worker in workers:
        data = worker.client_cache.get('data_key')
        features = data["test"]["features"]  # Or use a separate test key if available
        features = (features - features.min()) / (features.max() - features.min())

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(features)
            loss = criterion(reconstructed, features)
            total_loss += loss.item()

    avg_loss = total_loss / len(workers)
    print(f"Average Test Loss: {avg_loss:.4f}")
    return avg_loss


evaluate_global_model(global_autoencoder, workers)

# Evaluate and Visualize Results
print("\nEvaluating the global model...")
global_avg_loss = evaluate_global_model(global_autoencoder, workers)

print("\nPlotting reconstruction loss distribution...")
loss_data = plot_reconstruction_loss(global_autoencoder, workers)

print("\nComparing local and global model performance...")
local_global_comparison = evaluate_local_and_global(global_autoencoder, workers)

print("\nGenerating final report...")
generate_report(global_avg_loss, local_global_comparison)


threshold = define_threshold(global_autoencoder, workers)
precision, recall, f1 = evaluate_anomalies(global_autoencoder, workers, threshold)
plot_error_distributions(global_autoencoder, workers, threshold)
plot_precision_recall(global_autoencoder, workers)

