import torch
from flamby.datasets.fed_isic2019 import FedIsic2019
from torch.utils.data import DataLoader
from main import ViT, ViT_GPU
import flwr as fl
from opacus import PrivacyEngine
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import numpy as np
from plot_graphs import plot_metrics
import os

torch.manual_seed(2)
np.random.seed(2)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
client_name = "client_2"

if not os.path.exists(client_name):
    os.makedirs(client_name)

client_history = {
    "loss": [],
    "accuracy": [],
    "auc": []
}

PARAMS = {
    "batch_size": 32,
    "local_epochs": 3,
    "full_dataset_size": 18597,
    "number_of_classes": 8
}

PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "noise_multiplier": 0.41,
    "max_grad_norm": 1.0,
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Params: {PARAMS}")
print(f"Privacy Params: {PRIVACY_PARAMS}")


def save_str_to_file(string, dir: str):
    with open(f"{dir}/log_file.txt", "a") as file:
        file.write(string+'\n')


def load_data(client_index: int):
    train_dataset = FedIsic2019(center=client_index, train=True)
    test_dataset = FedIsic2019(train=False)
    trainloader = DataLoader(train_dataset, batch_size=PARAMS["batch_size"])
    testloader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"])
    sample_rate = len(train_dataset) / PARAMS["full_dataset_size"]
    return trainloader, testloader, sample_rate


def train(net, trainloader, privacy_engine, optimizer, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    
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
            optimizer.step()

            # Accumulate loss
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
        
    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    epsilon = privacy_engine.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
    return epsilon, average_loss


def test(net, testloader):
    """
    Evaluate the network on the test set.
    Returns (average_loss, accuracy, auc_score).
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    labels_list, scores_list = [], []
    correct, total_loss = 0, 0.0
    total_examples = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            scores = torch.softmax(outputs, dim=1)

            labels_list.append(labels.cpu().numpy())
            scores_list.append(scores.cpu().numpy())

            loss = criterion(outputs, labels).item()
            batch_size = images.size(0)
            total_loss += loss * batch_size
            total_examples += batch_size

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = correct / len(testloader.dataset)
    labels_array = np.concatenate(labels_list)
    scores_array = np.concatenate(scores_list)

    auc_score = roc_auc_score(
        y_true=labels_array,
        y_score=scores_array,
        labels=list(range(PARAMS["number_of_classes"])), 
        multi_class='ovr'
    )

    return average_loss, accuracy, auc_score


class FedViTDPClient2(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader) -> None:
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.testloader = testloader
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=model,
            optimizer=self.optimizer,
            data_loader=trainloader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=PRIVACY_PARAMS["noise_multiplier"],
        )

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
        epsilon, average_loss = train(
            self.model,
            self.trainloader,
            self.privacy_engine,
            self.optimizer,
            PARAMS["local_epochs"]
        )
        string = f"Epsilon: {epsilon:.2f}, Loss: {average_loss:.2f}"
        save_str_to_file(string, client_name)
        
        print(f"Epsilon = {epsilon:.2f}")
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"epsilon": epsilon}
        )

    def evaluate(self, parameters, config):
        """Evaluate the model using the provided parameters."""
        self.set_parameters(parameters)
        loss, accuracy, auc = test(self.model, self.testloader)
        client_history["loss"].append(loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc)
        string = f"Loss: {loss:.2f}, Accuracy: {accuracy:.2f}, AUC: {auc:.2f}"
        save_str_to_file(string, client_name)
        print(f"\n{client_history}\n")
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy, "auc": auc}

# model = ViT()
model = ViT_GPU(device=DEVICE)

trainload, testloader, sample_rate = load_data(client_index=1)
string = f"Train Dataset Size: {len(trainload.dataset)} Sample rate: {sample_rate}"
save_str_to_file(string, client_name)

fl.client.start_client(
    server_address="127.0.0.1:8725",
    client = FedViTDPClient2(model=model, trainloader=trainload, testloader=testloader).to_client()
)

plot_metrics(client_history, client_name)
print(f"\n\n{client_history}")