import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_metrics_from_csv(csv_file, client_name):
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Ensure client directory exists for saving files
    os.makedirs(client_name, exist_ok=True)

    epochs = data['Epoch']

    # Accuracy plot
    if 'Accuracy' in data:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, data['Accuracy'], 'b', label='Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        accuracy_path = os.path.join(client_name, 'accuracy_plot.png')
        plt.savefig(accuracy_path, dpi=300)
        plt.show()

    # AUC plot
    if 'AUC' in data:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, data['AUC'], 'r', label='AUC')
        plt.title('AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        auc_path = os.path.join(client_name, 'auc_plot.png')
        plt.savefig(auc_path, dpi=300)
        plt.show()

    # Loss plot
    if 'Loss' in data:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, data['Loss'], 'g', label='Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_path = os.path.join(client_name, 'loss_plot.png')
        plt.savefig(loss_path, dpi=300)
        plt.show()

# Example usage
csv_file = "client_1/log_file.csv"  # Replace with the path to your CSV file
client_name = "client_results"  # Directory to save the plots
plot_metrics_from_csv(csv_file, client_name)
