import os
import torch
import numpy as np
import flwr as fl
from opacus import PrivacyEngine
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import create_accountant

from flamby.datasets.fed_isic2019 import FedIsic2019
from main import ViT_GPU
from plot_graphs import plot_metrics


# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Select GPU and create a directory for logging
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
client_name = "client_4"
if not os.path.exists(client_name):
    os.makedirs(client_name)

# Client training history
client_history = {
    "loss": [],
    "accuracy": [],
    "auc": []
}

# Hyperparameters
PARAMS = {
    "batch_size": 32,
    "local_epochs": 3,
}

# Privacy parameters
PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "max_grad_norm": 1.0,
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_str_to_file(string: str, dir_path: str) -> None:
    """Append a string to the log file in the specified directory."""
    with open(f"{dir_path}/log_file.txt", "a") as file:
        file.write(string + '\n')


def load_data(client_index: int):
    """Load training and testing data for the given client index."""
    train_dataset = FedIsic2019(center=client_index, train=True)
    test_dataset = FedIsic2019(train=False)

    trainloader = DataLoader(train_dataset, batch_size=PARAMS["batch_size"])
    testloader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"])

    sample_rate = PARAMS["batch_size"] / len(train_dataset)
    return trainloader, testloader, sample_rate


def train(net, trainloader, privacy_engine, optimizer, epochs):
    """
    Train the network with Opacus for the specified number of epochs.
    Returns the final epsilon value and the last computed loss.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()  # <-- CRUCIAL: Update model parameters

    # A placeholder for epsilon (overwritten below or by external logic)
    epsilon = 30
    return epsilon, loss


def test(net, testloader):
    """
    Evaluate the network on the test set.
    Returns (total_loss, accuracy, auc_score).
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    labels_list, scores_list = [], []
    correct, total_loss = 0, 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            scores = torch.softmax(outputs, dim=1)

            labels_list.append(labels.cpu().numpy())
            scores_list.append(scores.cpu().numpy())
            total_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    labels_array = np.concatenate(labels_list)
    scores_array = np.concatenate(scores_list)

    auc_score = roc_auc_score(
        y_true=labels_array,
        y_score=scores_array,
        labels=list(range(8)),  # 8 classes
        multi_class='ovr'
    )

    return total_loss, accuracy, auc_score


def compute_fisher_information(model, dataloader, device):
    """
    Compute Fisher Information (diagonal approximation) for each parameter of the model.
    """
    fisher_diag = [torch.zeros_like(param).to(device) for param in model.parameters()]
    model.eval()

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        log_probs = F.log_softmax(outputs, dim=1)

        for i, label in enumerate(labels):
            log_prob = log_probs[i, label]
            model.zero_grad()
            log_prob.backward(retain_graph=True)

            for j, param in enumerate(model.parameters()):
                if param.grad is not None:
                    fisher_diag[j] += param.grad ** 2
            model.zero_grad()

    fisher_diag = [fisher_value / len(dataloader.dataset) for fisher_value in fisher_diag]
    return fisher_diag


def update_noise_multiplier(base_noise_multiplier, fisher_diag, client_data_size, avg_data_size, epoch, max_epochs):
        """
        Update the noise multiplier dynamically based on base noise, Fisher information, and client data scale.
        """
        # Fisher scaling: parameters with higher Fisher information should have lower noise
        #fisher_scaling_factor = ([1.0 / (f + 1e-6) for f in fisher_diag])  # Avoid division by zero
        fisher_scaling_factor = [f + 1e-6 for f in fisher_diag]
        print("BaseNM",base_noise_multiplier)

        # Client data scale: clients with more data need less noise
        total_size=24459
        data_scaling_factor = total_size / avg_data_size
       
        # Ensure that each element is a scalar or a float value
        fisher_scaling_factor = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in fisher_scaling_factor]

        # Iterate over each tensor in fisher_scaling_factor and apply operations individually
        for f in fisher_scaling_factor:
           noise_multiplier = base_noise_multiplier * f * data_scaling_factor

        # Combine base noise multiplier with Fisher scaling and data scaling
        #self.noise_multiplier = base_noise_multiplier * fisher_scaling_factor * data_scaling_factor
        noise_multiplier = noise_multiplier.tolist()  # Convert to list for each parameter
        
        # Ensure it's a scalar float
        if isinstance(noise_multiplier, list):
            noise_multiplier = sum(noise_multiplier) / len(noise_multiplier)  # Example: average if list
        
        if isinstance(noise_multiplier, torch.Tensor):
            noise_multiplier = noise_multiplier.item()  # Convert tensor to float if needed
        
        print("noise multiplier" , noise_multiplier)
        
        print(f"Noise multiplier before  adjustment: {noise_multiplier}")
             
        # Optional: Adjust noise based on some condition related to fisher scaling factor
        
        #noise_multiplier *= 0.9  # Apply less noise when fisher information is smaller (scale by 0.5)
        
        # Ensure the noise multiplier stays below 1.0
        #noise_multiplier = min(noise_multiplier, base_noise_multiplier)
        
        print(f"Noise multiplier before convergence adjustment: {noise_multiplier}")
        # Convergence-based adjustment
        # Reduce the noise multiplier as the model converges
        if epoch > max_epochs // 2:  # Starting to converge after half of the epochs
               # We reduce the noise multiplier gradually after halfway through the training
                convergence_factor = 1 - (epoch - max_epochs // 2) / (max_epochs // 2)
                noise_multiplier *= convergence_factor
       
        print(f"Updated noise multiplier after convergence adjustment: {noise_multiplier}")

        return noise_multiplier


class FedViTDPClient4(fl.client.NumPyClient):
    """
    Flower client for a Vision Transformer (ViT) model
    with differential privacy via Opacus.
    """
    def __init__(self, model, trainloader, testloader, sample_rate, fisher_diag) -> None:
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.sample_rate = sample_rate
        self.fisher_diag = fisher_diag
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.device = DEVICE
        self.fisher_threshold = 0.4
        self.lambda_1 = 0.05
        self.lambda_2 = 0.1
        self.clipping_bound = 2.4
        self.global_epoch = 1
        self.max_global_epochs = 30
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.privacy_engine = None

    def get_parameters(self, config):
        """Return the client's model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load the given parameters into the client's model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Perform local training (DP-enabled) using the provided parameters.
        Returns new parameters, number of examples, and a dict of metrics.
        """
        # 1) Update model parameters
        self.set_parameters(parameters)

        # 2) Compute dynamic noise multiplier
        client_data_size = len(self.trainloader.dataset)
        total_size = 23250
        avg_data_size = total_size / 6

        accountant = create_accountant(mechanism="prv")
        base_noise_multiplier = get_noise_multiplier(
            target_epsilon=30,
            target_delta=1e-5,
            sample_rate=1 / 6,
            epochs=PARAMS["local_epochs"],
            accountant=accountant.mechanism(),
        )
        dynamic_noise_multiplier = update_noise_multiplier(
            base_noise_multiplier,
            self.fisher_diag,
            client_data_size,
            avg_data_size,
            self.global_epoch,
            self.max_global_epochs
        )

        # 3) Re-initialize PrivacyEngine
        self.privacy_engine = PrivacyEngine()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        self.model.train()
        
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=dynamic_noise_multiplier
        )

        # 4) Perform local DP training
        epsilon, loss = train(
            self.model,
            self.trainloader,
            self.privacy_engine,
            self.optimizer,
            PARAMS["local_epochs"]
        )

        # Log training details
        log_str = f"Global Epoch (Round): {self.global_epoch}, Train Size: {len(self.trainloader.dataset)}, Sample Rate: {self.sample_rate}"
        save_str_to_file(log_str, client_name)
        print(f"Epsilon = {epsilon:.2f}")

        self.global_epoch += 1

        return (
            self.get_parameters(config={}),
            len(self.trainloader),
            {"epsilon": epsilon}
        )

    def evaluate(self, parameters, config):
        """
        Evaluate the model locally using the given parameters,
        and return the loss, number of examples, and metrics.
        """
        self.set_parameters(parameters)
        loss, accuracy, auc_score = test(self.model, self.testloader)

        client_history["loss"].append(loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc_score)

        log_str = f"Loss: {loss:.2f}, Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}"
        save_str_to_file(log_str, client_name)
        print(f"\n{client_history}\n")

        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    model = ViT_GPU(device=DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainload, testloader, sample_rate = load_data(client_index=3)

    init_log_str = f"Initial Train Dataset Size: {len(trainload.dataset)} Sample rate: {sample_rate}"
    save_str_to_file(init_log_str, client_name)

    fisher_diag = compute_fisher_information(model, trainload, device=device)
    client_data_size = len(trainload.dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    fl.client.start_client(
        server_address="127.0.0.1:8079",
        client=FedViTDPClient4(
            model=model,
            trainloader=trainload,
            testloader=testloader,
            sample_rate=sample_rate,
            fisher_diag=fisher_diag
        ).to_client()
    )

    plot_metrics(client_history, client_name)
    print(f"\n\nFinal client history:\n{client_history}\n")
