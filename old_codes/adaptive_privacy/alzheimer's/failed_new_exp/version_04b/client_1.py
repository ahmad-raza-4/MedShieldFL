import os
import copy
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

from main_new import ViT_GPU
from plot_graphs import plot_metrics

from torchvision import datasets, transforms

torch.manual_seed(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    "epsilon": 1.0,
    "target_epsilon": 1.0
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
    train_dir = f"/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/train/client_{client_index}/"
    test_dir = "/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/test"

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    testset = datasets.ImageFolder(root=test_dir, transform=transform)

    # --- Added code: Count the number of images per class in the training set ---
    from collections import Counter
    train_class_counts = Counter([s[1] for s in trainset.samples])
    save_str_to_file(f"Client {client_index} Train Class Counts: {train_class_counts}", client_name)
    for class_idx in range(PARAMS["number_of_classes"]):
        save_str_to_file(f"Client {client_index} Class {class_idx} Count: {train_class_counts[class_idx]}", client_name)

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
    print(f"Epochs: {epochs}, Trainloader Size: {len(trainloader.dataset)}\n")

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
    # epsilon = PRIVACY_PARAMS["epsilon"]
    epsilon = privacy_engine.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
    return epsilon, average_loss


# def test(net, testloader):
#     """
#     Evaluate the network on the test set.
#     Returns (average_loss, accuracy, auc_score).
#     """
#     criterion = torch.nn.CrossEntropyLoss()
#     net.eval()

#     labels_list, scores_list = [], []
#     correct, total_loss = 0, 0.0
#     total_examples = 0

#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
#             outputs = net(images)
#             scores = torch.softmax(outputs, dim=1)

#             labels_list.append(labels.cpu().numpy())
#             scores_list.append(scores.cpu().numpy())

#             loss = criterion(outputs, labels).item()
#             batch_size = images.size(0)
#             total_loss += loss * batch_size
#             total_examples += batch_size

#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == labels).sum().item()

#     average_loss = total_loss / total_examples if total_examples > 0 else 0.0
#     accuracy = correct / len(testloader.dataset)
#     labels_array = np.concatenate(labels_list)
#     scores_array = np.concatenate(scores_list)

#     auc_score = roc_auc_score(
#         y_true=labels_array,
#         y_score=scores_array,
#         labels=list(range(PARAMS["number_of_classes"])),
#         multi_class='ovr'
#     )

#     return average_loss, accuracy, auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def test(net, testloader):
    """
    Evaluate the network on the test set.
    Returns (average_loss, accuracy, auc_score, precision, recall, f1, confusion_matrix).
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    labels_list, scores_list, predictions_list = [], [], []
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
            predictions_list.append(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = correct / len(testloader.dataset)
    labels_array = np.concatenate(labels_list)
    scores_array = np.concatenate(scores_list)
    predictions_array = np.concatenate(predictions_list)

    auc_score = roc_auc_score(
        y_true=labels_array,
        y_score=scores_array,
        labels=list(range(PARAMS["number_of_classes"])),
        multi_class='ovr'
    )

    # Compute Confusion Matrix
    conf_matrix = confusion_matrix(labels_array, predictions_array)

    # Compute Precision, Recall, and F1 Score
    precision = precision_score(labels_array, predictions_array, average='macro', zero_division=0)
    recall = recall_score(labels_array, predictions_array, average='macro', zero_division=0)
    f1 = f1_score(labels_array, predictions_array, average='macro', zero_division=0)

    from collections import Counter
    pred_class_counts = Counter(predictions_array)
    true_class_counts = Counter(labels_array)
    save_str_to_file(f"Predicted Class Counts: {pred_class_counts}", client_name)
    for class_idx in range(PARAMS["number_of_classes"]):
        save_str_to_file(f"Class {class_idx} Count: {pred_class_counts[class_idx]}", client_name)
    save_str_to_file(f"True Class Counts: {true_class_counts}", client_name)
    for class_idx in range(PARAMS["number_of_classes"]):
        save_str_to_file(f"Class {class_idx} Count: {true_class_counts[class_idx]}", client_name)

    return average_loss, accuracy, auc_score, precision, recall, f1, conf_matrix


def compute_fisher_information(model, dataloader, device):
    """
    Compute the Fisher Information (diagonal approximation) for each parameter.
    """
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

    # Normalize by the total dataset size
    fisher_diag = [fisher_value / len(dataloader.dataset) for fisher_value in fisher_diag]

    # Normalize Fisher values layer-wise
    normalized_fisher_diag = []
    for fisher_value in fisher_diag:
        x_min = torch.min(fisher_value)
        x_max = torch.max(fisher_value)
        normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min)
        normalized_fisher_diag.append(normalized_fisher_value)

    return fisher_diag


def update_noise_multiplier(
    base_noise_multiplier, fisher_diag, client_data_size, epoch, max_epochs, epsilon
):
    # Convert Fisher tensors to scalars (mean of each parameter's Fisher values)
    fisher_scalars = []
    for f in fisher_diag:
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()
        fisher_scalars.append(np.mean(f))

    noise_multipliers = [
        base_noise_multiplier * f  
        for f in fisher_scalars
    ]

    # noise_multipliers = [base_noise_multiplier * (1.0 - f + 1e-8) for f in fisher_scalars]

    # Aggregate using mean
    noise_multiplier = np.mean(noise_multipliers)
    return float(noise_multiplier)


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
        # Fisher information will be computed dynamically in each epoch
        self.fisher_diag = fisher_diag
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.device = DEVICE
        self.fisher_threshold = 0.4
        self.lambda_1 = 0.1
        self.lambda_2 = 0.05
        self.clipping_bound = 0.5
        self.global_epoch = 1
        self.max_global_epochs = 30
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.privacy_engine = None
        self.epsilon=PRIVACY_PARAMS["epsilon"]

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

        clean_model = ViT_GPU(device=self.device)
        # Adjust the state dict: remove DP-induced prefixes if present
        state_dict = self.model.state_dict()
        new_state_dict = {}
        prefix = "_module."
        for key, value in state_dict.items():
            new_key = key[len(prefix):] if key.startswith(prefix) else key
            new_state_dict[new_key] = value
        clean_model.load_state_dict(new_state_dict)
        clean_model.to(self.device)
        
        print(f"Step 1.5: Recomputing Fisher Information dynamically for Global Epoch {self.global_epoch}")
        self.fisher_diag = compute_fisher_information(clean_model, self.trainloader, self.device)
    
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
        
        # epsilon = self.privacy_engine.accountant.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
        # epsilon = self.privacy_engine.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
        # save_str_to_file(f"Updated Epsilon: {epsilon:.4f}", client_name)

        dynamic_noise_multiplier = update_noise_multiplier(
            base_noise_multiplier=base_noise_multiplier,
            fisher_diag=self.fisher_diag,
            client_data_size=client_data_size,
            epoch=self.global_epoch,
            max_epochs=self.max_global_epochs,
            # epsilon=PRIVACY_PARAMS["epsilon"]
            epsilon=self.epsilon
        )

        print(f"\nBase NM: {base_noise_multiplier:.4f}, Dynamic NM: {dynamic_noise_multiplier:.4f}\n")

        print("Step 2c: Re-initialize PrivacyEngine")
        # 3. Re-initialize PrivacyEngine
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

        self.epsilon = epsilon

        print("Step 2f: Log training details")
        log_str = (
            f"Global Epoch (Round): {self.global_epoch}, "
            f"Train Size: {len(self.trainloader.dataset)}, "
            f"Sample Rate: {self.sample_rate}, "
            f"Train Loss: {average_loss:.4f}, "
            f"Epsilon: {epsilon:.4f}, "
            f"Base Noise Multiplier: {base_noise_multiplier:.4f}, "
            f"Dynamic Noise Multiplier: {dynamic_noise_multiplier:.4f}"
        )
        save_str_to_file(log_str, client_name)

        self.global_epoch += 1

        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"epsilon": epsilon, "train_loss": average_loss}
        )

    # def evaluate(self, parameters, config):
    #     """
    #     Evaluate the model locally using the given parameters,
    #     and return the loss, number of examples, and metrics.
    #     """
    #     print("Step 3: Evaluate the model locally")
    #     self.set_parameters(parameters)
    #     average_loss, accuracy, auc_score = test(self.model, self.testloader)

    #     client_history["loss"].append(average_loss)
    #     client_history["accuracy"].append(accuracy)
    #     client_history["auc"].append(auc_score)

    #     log_str = (
    #         f"Loss: {average_loss:.4f}, "
    #         f"Accuracy: {accuracy:.4f}, "
    #         f"AUC: {auc_score:.4f}"
    #     )
    #     save_str_to_file(log_str, client_name)
    #     print(f"\n{client_history}\n")

    #     return float(average_loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        """
        Evaluate the model locally using the given parameters,
        and return the loss, number of examples, and metrics.
        """
        print("Step 3: Evaluate the model locally")
        self.set_parameters(parameters)
        
        # Get all evaluation metrics
        average_loss, accuracy, auc_score, precision, recall, f1, conf_matrix = test(self.model, self.testloader)

        client_history["loss"].append(average_loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc_score)

        # Convert Confusion Matrix to String Format
        conf_matrix_str = "\n".join(["\t".join(map(str, row)) for row in conf_matrix])

        log_str = (
            f"Loss: {average_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"AUC: {auc_score:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1 Score: {f1:.4f}\n"
            f"Confusion Matrix:\n{conf_matrix_str}\n"
        )

        save_str_to_file(log_str, client_name)
        print(f"\n{client_history}\n")

        return float(average_loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc_score)
        }


if __name__ == "__main__":
    model = ViT_GPU(device=DEVICE)
    trainload, testloader, sample_rate = load_data(client_index=0)

    init_log_str = f"Initial Train Dataset Size: {len(trainload.dataset)} Sample rate: {sample_rate}"
    save_str_to_file(init_log_str, client_name)

    fisher_diag = compute_fisher_information(model, trainload, DEVICE)

    # Start the Flower client (Fisher Information is now computed dynamically in each round)
    fl.client.start_client(
        server_address="127.0.0.1:8069",
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
