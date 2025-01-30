import matplotlib.pyplot as plt

def plot_metrics(history, client_name):
    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy plot
    if(len(history['accuracy'])):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['accuracy'], 'b', label='Accuracy')
        plt.title(f'{client_name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{client_name}/accuracy_plot.png', dpi=300)
        plt.show()

    # AUC plot
    if(len(history['auc'])):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['auc'], 'r', label='AUC')
        plt.title(f'{client_name} AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{client_name}/auc_plot.png', dpi=300)
        plt.show()

    # Loss plot
    if(len(history['loss'])):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['loss'], 'g', label='Loss')
        plt.title(f'{client_name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{client_name}/loss_plot.png', dpi=300)
        plt.show()
