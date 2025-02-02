
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from model.autoencoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


# Visualize Reconstruction Loss Distributions
def plot_reconstruction_loss(global_model, global_scaler, workers, data_key="value"):
    import torch.nn as nn
    criterion = nn.MSELoss()
    losses = []
    for worker in workers:
        data = worker.client_cache.get('data_key')
        features = data["test"]["features"]  # Or use a separate test key if available
        # **Use global scaler**
        features = global_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(features)
            loss = criterion(reconstructed, features)
            losses.append(loss.item())

    # Create a DataFrame for visualization
    loss_data = pd.DataFrame({
        "Worker": [worker.name for worker in workers],
        "Reconstruction Loss": losses
    })

    # Plot the results
    sns.barplot(x="Worker", y="Reconstruction Loss", data=loss_data)
    plt.title("Reconstruction Loss Across Workers")
    plt.xlabel("Worker")
    plt.ylabel("Loss")
    plt.show()

    return loss_data


# Compare Local and Global Model Performance
def evaluate_local_and_global(global_model, global_scaler, workers, data_key="value"):
    import torch.nn as nn
    global_criterion = nn.MSELoss()
    results = []

    for worker in workers:
        data = worker.client_cache.get('data_key')
        features = data["test"]["features"]
        # **Use global scaler**
        features = global_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Global model evaluation
        global_model.eval()
        with torch.no_grad():
            global_reconstructed = global_model(features)
            global_loss = global_criterion(global_reconstructed, features).item()

        # Local model evaluation
        local_state = worker.client_cache.get('local_value')
        local_autoencoder = AutoEncoder(output_dim=features.shape[1])
        local_autoencoder.load_state_dict(local_state)
        local_autoencoder.eval()
        with torch.no_grad():
            local_reconstructed = local_autoencoder(features)
            local_loss = global_criterion(local_reconstructed, features).item()

        results.append({
            "Worker": worker.name,
            "Global Loss": global_loss,
            "Local Loss": local_loss
        })

    # Create a DataFrame for comparison
    results_df = pd.DataFrame(results)

    # Plot Global vs Local Loss
    results_df.set_index("Worker")[["Global Loss", "Local Loss"]].plot(kind="bar")
    plt.title("Global vs Local Reconstruction Loss")
    plt.ylabel("Loss")
    plt.show()

    return results_df


# Generate Tables and Final Report
def generate_report(global_loss, local_global_comparison):
    print("\n### Anomaly Detection Results ###")
    print("\nGlobal Model Evaluation:")
    print(f"Average Global Test Loss: {global_loss:.4f}\n")

    print("\nLocal vs Global Model Comparison:")
    print(local_global_comparison.to_string(index=False))



def define_threshold(global_model, global_scaler, workers, data_key="value"):
    """
    Define anomaly detection threshold using reconstruction errors on benign traffic.
    """
    criterion = nn.MSELoss()
    benign_errors = []

    for worker in workers:
        data = worker.client_cache.get('data_key')
        features = data['train']['features']
        # **Use global scaler**
        features = global_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(features)
            loss = criterion(reconstructed, features)
            benign_errors.append(loss.item())

    # Threshold based on mean + 3 * std of benign reconstruction errors
    benign_errors = torch.tensor(benign_errors)
    threshold = benign_errors.mean().item() + 1 * benign_errors.std().item()
    print(f"Defined Threshold: {threshold:.4f}")
    return threshold


def get_threshold_global(global_model, args, device_list):
    th_local_dict = dict()
    min_max_file_path = "/Users/stefanbehfar/Documents/Projects/FedML/iot/anomaly_detection_for_cybersecurity/data"
    min_dataset = np.loadtxt(os.path.join(min_max_file_path, "min_dataset.txt"))
    max_dataset = np.loadtxt(os.path.join(min_max_file_path, "max_dataset.txt"))

    for i, device_name in enumerate(device_list):
        benign_data = pd.read_csv(
            os.path.join(args.data_cache_dir, device_name, "benign_traffic.csv")
        ).values  # Convert DataFrame to NumPy

        benign_th = benign_data[5000:8000]
        benign_th[np.isnan(benign_th)] = 0  # Replace NaNs with 0

        # Avoid division by zero in normalization
        benign_th = (benign_th - min_dataset) / (max_dataset - min_dataset + 1e-8)

        # Ensure correct dtype for PyTorch
        benign_th = torch.tensor(benign_th, dtype=torch.float32)

        th_local_dict[i] = torch.utils.data.DataLoader(
            benign_th, batch_size=128, shuffle=False, num_workers=0
        )

    model = global_model
    model.eval()

    mse = []
    threshold_func = nn.MSELoss(reduction="none")

    for client_index, train_data in th_local_dict.items():
        for batch_idx, x in enumerate(train_data):
            x = x.to(torch.float32)  # Ensure float32 consistency
            with torch.no_grad():  # Avoid gradient tracking
                diff = threshold_func(model(x), x)
            mse.append(diff)

    if len(mse) == 0:
        raise ValueError("No MSE values were computed. Check your DataLoader.")

    # Concatenate tensors safely
    mse_global = torch.cat(mse, dim=0).mean(dim=1)

    # Compute threshold (Mean + 3 * StdDev)
    threshold_global = torch.mean(mse_global) + 3 * torch.std(mse_global)

    return threshold_global



def evaluate_anomalies(global_model, global_scaler, workers, threshold, data_key="value"):
    """
    Evaluate detection of anomalies based on the defined threshold.
    """
    y_true = []
    y_pred = []

    for worker in workers:
        data = worker.client_cache.get('data_key')
        features = data["test"]["features"]
        labels = data["test"]["labels"]  # Ground truth labels
        # **Use global scaler**
        features = global_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(features)
            reconstruction_errors = ((reconstructed - features) ** 2).mean(dim=1)

        # Predict anomalies (1 if error > threshold, 0 otherwise)
        predictions = (reconstruction_errors > threshold).int()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return precision, recall, f1




def plot_error_distributions(global_model, global_scaler, workers, threshold, data_key="value"):
    """
    Plot reconstruction error distributions for benign and attack samples.
    """
    benign_errors = []
    attack_errors = []

    for worker in workers:
        data = worker.client_cache.get('data_key')
        features = data["test"]["features"]
        labels = data["test"]["labels"]
        # **Use global scaler**
        features = global_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(features)
            reconstruction_errors = ((reconstructed - features) ** 2).mean(dim=1)

        # Separate errors for benign and attack samples
        benign_errors.extend(reconstruction_errors[labels == 0].tolist())
        attack_errors.extend(reconstruction_errors[labels == 1].tolist())

    # Plot histograms
    plt.hist(benign_errors, bins=50, alpha=0.7, label="Benign")
    plt.hist(attack_errors, bins=50, alpha=0.7, label="Attack")
    plt.axvline(x=threshold, color="red", linestyle="--", label="Threshold")

    # Set x-axis limit to 20
    plt.xlim(0, 20)

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.show()


def plot_precision_recall(global_model, global_scaler, workers, data_key="value"):
    """
    Plot the precision-recall curve to visualize detection performance.
    """
    y_true = []
    y_scores = []

    for worker in workers:
        data = worker.client_cache.get('data_key')
        features = data["test"]["features"]
        labels = data["test"]["labels"]
        # **Use global scaler**
        features = global_scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        global_model.eval()
        with torch.no_grad():
            reconstructed = global_model(features)
            reconstruction_errors = ((reconstructed - features) ** 2).mean(dim=1)

        y_true.extend(labels.tolist())
        y_scores.extend(reconstruction_errors.tolist())

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Plot the curve
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
