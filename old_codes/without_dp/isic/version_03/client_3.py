import os
import torch
from flamby.datasets.fed_isic2019 import FedIsic2019
from torch.utils.data import DataLoader
from main import ViT, ViT_GPU  # ensure that ViT_GPU is defined in your main module
import flwr as fl
from collections import OrderedDict
import numpy as np
from plot_graphs import plot_metrics

# Additional metric imports
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

# For reproducibility
torch.manual_seed(3)
np.random.seed(3)

# Set CUDA device if available
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
client_name = "client_3"
if not os.path.exists(client_name):
    os.makedirs(client_name)

client_history = {
    "loss": [],
    "accuracy": [],
    "auc": [],
    # New metrics will be added to the log
    "balanced_accuracy": [],
    "f1_score": [],
    "confusion_matrix": []  # We'll store the confusion matrix as a list (for logging)
}

PARAMS = {
    "batch_size": 32,
    "local_epochs": 3,
    "full_dataset_size": 18597,
    "number_of_classes": 8
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Client Name: {client_name}")
print(f"Params: {PARAMS}")


def save_str_to_file(string, dir_path: str):
    with open(f"{dir_path}/log_file.txt", "a") as file:
        file.write(string + '\n')


def load_data(client_index: int):
    train_dataset = FedIsic2019(center=client_index, train=True)
    test_dataset = FedIsic2019(train=False)
    trainloader = DataLoader(train_dataset, batch_size=PARAMS["batch_size"])
    testloader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"])
    sample_rate = len(train_dataset) / PARAMS["full_dataset_size"]
    return trainloader, testloader, sample_rate


def test(net, testloader):
    """
    Evaluate the network on the test set.
    Returns:
        average_loss, accuracy, auc_score, balanced_accuracy, f1_score, confusion_matrix
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    labels_list = []
    scores_list = []
    predictions_list = []
    correct, total_loss = 0, 0.0
    total_examples = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            scores = torch.softmax(outputs, dim=1)

            # Append raw labels and scores for AUC computation
            labels_list.append(labels.cpu().numpy())
            scores_list.append(scores.cpu().numpy())

            loss = criterion(outputs, labels).item()
            batch_size = images.size(0)
            total_loss += loss * batch_size
            total_examples += batch_size

            # Get predictions and store them for other metrics
            _, predicted = torch.max(outputs.data, 1)
            predictions_list.append(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = correct / len(testloader.dataset)

    # Concatenate arrays from all batches
    labels_array = np.concatenate(labels_list)
    scores_array = np.concatenate(scores_list)
    predictions_array = np.concatenate(predictions_list)

    # Compute multi-class AUC (One-vs-Rest)
    auc_score = roc_auc_score(
        y_true=labels_array,
        y_score=scores_array,
        labels=list(range(PARAMS["number_of_classes"])),
        multi_class='ovr'
    )
    
    # Compute balanced accuracy (averages recall per class)
    balanced_acc = balanced_accuracy_score(labels_array, predictions_array)
    
    # Compute macro F1 score (average F1 score across classes)
    f1 = f1_score(labels_array, predictions_array, average='macro')
    
    # Compute confusion matrix
    cm = confusion_matrix(labels_array, predictions_array)
    
    return average_loss, accuracy, auc_score, balanced_acc, f1, cm


class FedViTClient3(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader) -> None:
        super().__init__()
        self.model = model
        self.testloader = testloader
        self.trainloader = trainloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        """Get the locally updated parameters in NumPy format."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model using the provided parameters and return the updated parameters."""
        self.set_parameters(parameters)
        
        average_loss = train(
            net=self.model,
            trainloader=self.trainloader,
            optimizer=self.optimizer,
            epochs=PARAMS["local_epochs"]
        )

        log_str = f"Average Loss: {average_loss:.4f}"
        save_str_to_file(log_str, client_name)
        
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"train_loss": average_loss}
        )

    def evaluate(self, parameters, config):
        """Evaluate the model using the provided parameters."""
        self.set_parameters(parameters)

        (loss,
         accuracy,
         auc,
         balanced_acc,
         f1,
         cm) = test(self.model, self.testloader)

        client_history["loss"].append(loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc)
        client_history["balanced_accuracy"].append(balanced_acc)
        client_history["f1_score"].append(f1)
        client_history["confusion_matrix"].append(cm.tolist())

        log_str = (
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, "
            f"Balanced Acc: {balanced_acc:.4f}, F1 Score: {f1:.4f}\n"
            f"Confusion Matrix:\n{cm}"
        )
        save_str_to_file(log_str, client_name)
        print(f"\n{client_history}\n")
        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
            "balanced_accuracy": balanced_acc,
            "f1_score": f1
        }


def train(net, trainloader, optimizer, epochs):
    """
    Train the network for the specified number of epochs.
    Returns the average loss.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    print("Training the model with the following parameters:")
    print(f"Epochs: {epochs}, Trainloader Size: {len(trainloader.dataset)}, Optimizer: {optimizer}\n")

    total_loss = 0.0
    total_examples = 0

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if images.size(0) == 0:
                continue  # Skip empty batches
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    return average_loss


if __name__ == "__main__":
    model = ViT_GPU(device=DEVICE)

    trainloader, testloader, sample_rate = load_data(client_index=2)
    log_str = f"Train Dataset Size: {len(trainloader.dataset)} Sample rate: {sample_rate}"
    save_str_to_file(log_str, client_name)

    # Start the Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8067",
        client=FedViTClient3(
            model=model, 
            trainloader=trainloader, 
            testloader=testloader
        )
    )

    # Optionally, plot your tracked metrics
    plot_metrics(client_history, client_name)
    print(f"\n\nFinal client history:\n{client_history}\n")
