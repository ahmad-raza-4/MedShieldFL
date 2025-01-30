import os
import torch
import numpy as np
import flwr as fl
from opacus import PrivacyEngine
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import create_accountant

from main import ViT_GPU
from plot_graphs import plot_metrics

from torchvision import datasets, transforms

torch.manual_seed(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
client_name = "client_1"
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
    "full_dataset_size": 6400,
    "number_of_classes": 4
}

PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "max_grad_norm": 1.0,
    "epsilon": 30,
    "target_epsilon": 30
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Params:\n batch_size: {PARAMS['batch_size']}, local_epochs: {PARAMS['local_epochs']}, full_dataset_size: {PARAMS['full_dataset_size']}, num_classes: {PARAMS['number_of_classes']}\n")
print(f"Privacy Params:\n epsilon: {PRIVACY_PARAMS['epsilon']}, target_epsilon: {PRIVACY_PARAMS['target_epsilon']}, target_delta: {PRIVACY_PARAMS['target_delta']}\n")
print(f"Device: {DEVICE}\n")


# Helper function to save a string to a file
def save_str_to_file(string: str, dir_path: str) -> None:
    """Append a string to the log file in the specified directory."""
    with open(f"{dir_path}/log_file.txt", "a") as file:
        file.write(string + '\n')


# Helper function to load training and testing data
def load_data(client_index: int):
    """Load training and testing data for the given client index."""
    # Paths to the training and testing directories
    train_dir = f"/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/train/client_{client_index}/"
    test_dir = "/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/test"

    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor() 
    ])

    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    testset = datasets.ImageFolder(root=test_dir, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=PARAMS["batch_size"], shuffle=True,
        num_workers=0,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=PARAMS["batch_size"], shuffle=False,
        num_workers=0
    )

    sample_rate = len(trainset) / PARAMS["full_dataset_size"]
    return trainloader, testloader, sample_rate


def train(net, trainloader, privacy_engine, optimizer, epochs):
    """
    Train the network with Opacus for the specified number of epochs.
    Returns the final epsilon value and the average loss.
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
            optimizer.step()

            # Accumulate loss
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    epsilon = PRIVACY_PARAMS["epsilon"]
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


def compute_fisher_information(model, dataloader, device):
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]
    model.train()

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        log_probs = F.log_softmax(outputs, dim=1)
        model.zero_grad()

        # Compute gradients for each sample
        for i in range(len(labels)):
            log_prob = log_probs[i, labels[i]]
            log_prob.backward(retain_graph=True)

            # Accumulate squared gradients
            with torch.no_grad():
                for j, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        fisher_diag[j] += param.grad ** 2
            model.zero_grad()

    # Normalize by total dataset size
    fisher_diag = [fisher_value / len(dataloader.dataset) for fisher_value in fisher_diag]
    return fisher_diag

def update_noise_multiplier(
    base_noise_multiplier, fisher_diag, client_data_size, epoch, max_epochs, epsilon
):
    # Convert Fisher tensors to scalars (mean of each parameter's Fisher values)
    fisher_scalars = []
    for f in fisher_diag:
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()
        fisher_scalars.append(np.mean(f))  # Reduce tensor to scalar

    # Compute per-parameter noise multipliers
    data_scaling_factor = client_data_size / PARAMS["full_dataset_size"]
    noise_multipliers = [
        base_noise_multiplier * f * data_scaling_factor 
        for f in fisher_scalars  # Now a list of scalars
    ]

    # Aggregate using mean
    noise_multiplier = np.mean(noise_multipliers)
    return float(noise_multiplier)


# # Helper function to compute Fisher Information which is used to compute dynamic noise multiplier
# def compute_fisher_information(model, dataloader, device):
#     """
#     Compute Fisher Information (diagonal approximation) for each parameter of the model.
#     """
#     fisher_diag = [torch.zeros_like(param).to(device) for param in model.parameters()]
#     model.train()

#     for data, labels in dataloader:
#         data, labels = data.to(device), labels.to(device)
#         outputs = model(data)
#         log_probs = F.log_softmax(outputs, dim=1)

#         for i, label in enumerate(labels):
#             log_prob = log_probs[i, label]
#             model.zero_grad()
#             log_prob.backward(retain_graph=True)

#             for j, param in enumerate(model.parameters()):
#                 if param.grad is not None:
#                     fisher_diag[j] += param.grad ** 2
#             model.zero_grad()

#     fisher_diag = [fisher_value / len(dataloader.dataset) for fisher_value in fisher_diag]
#     return fisher_diag


# # Helper function to update noise multiplier dynamically based on Fisher Information
# def update_noise_multiplier(base_noise_multiplier, fisher_diag, client_data_size, epoch, max_epochs, epsilon):
#         """
#         Update the noise multiplier dynamically based on base noise, Fisher information, and client data scale.
#         """
#         fisher_scaling_factor = [f + 1e-6 for f in fisher_diag]
#         print("Base Noise Multiplier Received: ",base_noise_multiplier)

        
#         data_scaling_factor = client_data_size / PARAMS["full_dataset_size"]
#         print(f"Data Scaling Factor: {data_scaling_factor} where Client Data Size: {client_data_size}")
       
#         fisher_scaling_factor = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in fisher_scaling_factor]

#         for f in fisher_scaling_factor:
#            noise_multiplier = base_noise_multiplier * f * data_scaling_factor

#         noise_multiplier = noise_multiplier.tolist()  
#         print("Noise Multiplier after Fisher Scaling: ",noise_multiplier)

#         if isinstance(noise_multiplier, list):
#             noise_multiplier = sum(noise_multiplier) / len(noise_multiplier) 
    
#         if isinstance(noise_multiplier, torch.Tensor):
#             noise_multiplier = noise_multiplier.item()

#         print("Noise Multiplier after list and tensor: " , noise_multiplier)
        
#         # noise_multiplier *= (1 / epsilon)  
        
#         # print("Noise Multiplier after Epsilon Scaling: ",noise_multiplier)
        
#         # # convergence factor
#         # if epoch > max_epochs // 2:
#         #         convergence_factor = 1 - (epoch - max_epochs // 2) / (max_epochs // 2)
#         #         noise_multiplier *= convergence_factor
       
#         # print(f"Noise Multiplier after Convergence: {noise_multiplier}")

#         return noise_multiplier


class FedViTDPClient1(fl.client.NumPyClient):
    """
    Flower client for a Vision Transformer (ViT) model with differential privacy via Opacus.
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.privacy_engine = None

        print("Step 1: Client Initialized")


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

        print("Step 2a: Compute base noise multiplier")

        accountant = create_accountant(mechanism="prv")
        base_noise_multiplier = get_noise_multiplier(
            target_epsilon=PRIVACY_PARAMS["target_epsilon"],
            target_delta=PRIVACY_PARAMS["target_delta"],
            sample_rate=client_data_size / PARAMS["full_dataset_size"],
            epochs=PARAMS["local_epochs"],
            accountant=accountant.mechanism(),
        )

        print("Step 2b: Update noise multiplier dynamically")

        dynamic_noise_multiplier = update_noise_multiplier(
            base_noise_multiplier=base_noise_multiplier,
            fisher_diag=self.fisher_diag,
            client_data_size=client_data_size,
            epoch=self.global_epoch,
            max_epochs=self.max_global_epochs,
            epsilon=PRIVACY_PARAMS["epsilon"]
        )

        print("Step 2c: Re-initialize PrivacyEngine")

        # 3) Re-initialize PrivacyEngine
        self.privacy_engine = PrivacyEngine()

        print("Step 2d: Make model private")

        self.model.train()
        
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=dynamic_noise_multiplier
        )

        print("Step 2e: Perform local DP training")

        # 4) Perform local DP training
        epsilon, average_loss = train(
            self.model,
            self.trainloader,
            self.privacy_engine,
            self.optimizer,
            PARAMS["local_epochs"]
        )

        print("Step 2f: Log training details")

        # Log training details with average loss
        log_str = (
            f"Global Epoch (Round): {self.global_epoch}, "
            f"Train Size: {len(self.trainloader.dataset)}, "
            f"Sample Rate: {self.sample_rate}, "
            f"Train Loss: {average_loss:.4f}, "
            f"Epsilon: {epsilon:.2f}, "
            f"Dynamic Noise Multiplier: {dynamic_noise_multiplier:.2f}"
        )
        save_str_to_file(log_str, client_name)

        self.global_epoch += 1

        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"epsilon": epsilon, "train_loss": average_loss}
        )


    def evaluate(self, parameters, config):
        """
        Evaluate the model locally using the given parameters,
        and return the loss, number of examples, and metrics.
        """

        print("Step 3: Evaluate the model locally")

        self.set_parameters(parameters)
        average_loss, accuracy, auc_score = test(self.model, self.testloader)

        client_history["loss"].append(average_loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc_score)

        log_str = (
            f"Loss: {average_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"AUC: {auc_score:.4f}"
        )
        save_str_to_file(log_str, client_name)
        print(f"\n{client_history}\n")

        return float(average_loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}



if __name__ == "__main__":
    model = ViT_GPU(device=DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainload, testloader, sample_rate = load_data(client_index=0)

    init_log_str = f"Initial Train Dataset Size: {len(trainload.dataset)} Sample rate: {sample_rate}"
    save_str_to_file(init_log_str, client_name)

    fisher_diag = compute_fisher_information(model, trainload, device=device)
    client_data_size = len(trainload.dataset)

    fl.client.start_client(
        server_address="127.0.0.1:8013",
        client=FedViTDPClient1(
            model=model,
            trainloader=trainload,
            testloader=testloader,
            sample_rate=sample_rate,
            fisher_diag=fisher_diag
        ).to_client()
    )

    plot_metrics(client_history, client_name)
    print(f"\n\nFinal client history:\n{client_history}\n")
