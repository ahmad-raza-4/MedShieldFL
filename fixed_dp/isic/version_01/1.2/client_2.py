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

torch.manual_seed(1)
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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
}

PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "noise_multiplier": 1.2,
    "max_grad_norm": 1.0,
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_str_to_file(string, dir: str):
    with open(f"{dir}/log_file.txt", "a") as file:
        file.write(string+'\n')

def load_data(client_index: int):
    train_dataset = FedIsic2019(center=client_index, train=True)
    test_dataset = FedIsic2019(train=False)
    trainloader = DataLoader(train_dataset, batch_size=PARAMS["batch_size"])
    testloader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"])
    sample_rate = PARAMS["batch_size"] / len(train_dataset)
    return trainloader, testloader, sample_rate

def train(net, trainloader, privacy_engine, optimizer, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    epsilon = privacy_engine.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
    save_str_to_file(f"Epsilon = {epsilon:.2f}", client_name)
    return epsilon

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    labels_list, scores_list = [], [] # Add 1

    correct, loss = 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)

            # Add 2
            scores = torch.softmax(outputs, dim=1)
            labels_list.append(labels.cpu().numpy())
            scores_list.append(scores.cpu().numpy())

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)

    # Add 3
    labels_array = np.concatenate(labels_list)
    scores_array = np.concatenate(scores_list)
    auc_score = roc_auc_score(y_true=labels_array, y_score=scores_array, multi_class='ovr')

    return loss, accuracy, auc_score


class FedViTDPClient2(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader) -> None:
        super().__init__()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        self.testloader = testloader
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
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
        epsilon = train(
            self.model,
            self.trainloader,
            self.privacy_engine,
            self.optimizer,
            PARAMS["local_epochs"]
        )
        string = f"Train Dataset Size: {len(self.trainloader)} Sample rate: {sample_rate}"
        save_str_to_file(string, client_name)
        
        print(f"Epsilon = {epsilon:.2f}")
        return (
            self.get_parameters(config={}),
            len(self.trainloader),
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
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

# model = ViT()
model = ViT_GPU(device=DEVICE)

trainload, testloader, sample_rate = load_data(client_index=1)
string = f"Train Dataset Size: {len(trainload)} Sample rate: {sample_rate}"
save_str_to_file(string, client_name)

fl.client.start_client(
    server_address="127.0.0.1:8084",
    client = FedViTDPClient2(model=model, trainloader=trainload, testloader=testloader).to_client()
)

plot_metrics(client_history, client_name)
print(f"\n\n{client_history}")