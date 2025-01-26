import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for CSV operations
import os

def plot_metrics(history, client_name):
    epochs = range(1, len(history['accuracy']) + 1)

    # Ensure client directory exists for saving files
    os.makedirs(client_name, exist_ok=True)

    # Accuracy plot
    if len(history['accuracy']):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['accuracy'], 'b', label='Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        accuracy_path = os.path.join(client_name, 'accuracy_plot.png')
        plt.savefig(accuracy_path, dpi=300)
        plt.show()

        # Save CSV for accuracy
        try:
            df_acc = pd.DataFrame({
                'Epoch': list(epochs),
                'Accuracy': history['accuracy']
            })
            csv_acc_path = os.path.join(client_name, 'accuracy_data.csv')
            df_acc.to_csv(csv_acc_path, index=False)
        except Exception as e:
            print(f"Failed to save accuracy CSV: {e}")

    # AUC plot
    if len(history['auc']):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['auc'], 'r', label='AUC')
        plt.title('AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        auc_path = os.path.join(client_name, 'auc_plot.png')
        plt.savefig(auc_path, dpi=300)
        plt.show()

        # Save CSV for AUC
        try:
            df_auc = pd.DataFrame({
                'Epoch': list(epochs),
                'AUC': history['auc']
            })
            csv_auc_path = os.path.join(client_name, 'auc_data.csv')
            df_auc.to_csv(csv_auc_path, index=False)
        except Exception as e:
            print(f"Failed to save AUC CSV: {e}")

    # Loss plot
    if len(history['loss']):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['loss'], 'g', label='Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_path = os.path.join(client_name, 'loss_plot.png')
        plt.savefig(loss_path, dpi=300)
        plt.show()

        # Save CSV for Loss
        try:
            df_loss = pd.DataFrame({
                'Epoch': list(epochs),
                'Loss': history['loss']
            })
            csv_loss_path = os.path.join(client_name, 'loss_data.csv')
            df_loss.to_csv(csv_loss_path, index=False)
        except Exception as e:
            print(f"Failed to save Loss CSV: {e}")
