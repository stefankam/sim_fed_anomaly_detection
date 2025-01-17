import torch
import syft as sy
from syft_simulate import load_device_data
from model.autoencoder import AutoEncoder
from torch import nn

# Step 1: Prepare Datasets (only run once)
data_cache_dir = "./data_cache"
#prepare_datasets(data_cache_dir)

# Step 2: Create Workers
device_names = ["Danmini_Doorbell", "Ecobee_Thermostat"]
workers = [sy.Worker(name=name) for name in device_names]
server = sy.Worker(name="server")

# Step 3: Load Data for Each Device
for worker in workers:
    features, labels = load_device_data(worker.name, data_cache_dir)
    device_data = {"train": {"features": features, "labels": labels}}
    worker.client_cache.update(value=device_data, description=f"Data for {worker.name}")

# Step 4: Train AutoEncoders
def train_local_autoencoder(worker, data_key, output_dim=115, epochs=5, lr=0.001, batch_size=32):
    # Retrieve data from the client cache
    #data = worker.client_cache.get(data_key)
    features = device_data['train']['features']

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

num_rounds = 5
output_dim = 115
epochs_per_round = 5
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
        local_update = train_local_autoencoder(worker, "value", output_dim=output_dim, epochs=epochs_per_round, lr=lr)
        local_updates.append(local_update)

    # Step 3: Aggregate Updates
    global_state = aggregate_updates(local_updates, global_autoencoder)
    global_autoencoder.load_state_dict(global_state)

    print(f"Round {round_num}: Global model updated and distributed.")

print("\nFederated learning completed.")



def evaluate_global_model(global_model, workers, data_key="value"):
    total_loss = 0
    criterion = nn.MSELoss()

    for worker in workers:
        #data = worker.client_cache.get(data_key)
        test_features = device_data["train"]["features"]  # Or use a separate test key if available

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(test_features)
            loss = criterion(reconstructed, test_features)
            total_loss += loss.item()

    avg_loss = total_loss / len(workers)
    print(f"Average Test Loss: {avg_loss:.4f}")
    return avg_loss


evaluate_global_model(global_autoencoder, workers)