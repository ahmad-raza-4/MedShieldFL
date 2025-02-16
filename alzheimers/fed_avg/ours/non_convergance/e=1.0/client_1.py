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

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
client_name = "client_1"
if not os.path.exists(client_name):
    os.makedirs(client_name)

client_history = {
    "loss": [],
    "accuracy": [],
    "auc": [],
    "precision": [],
    "recall": [],
    "f1": []
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
    "epsilon": 1,
    "target_epsilon": 1
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Params:\n batch_size: {PARAMS['batch_size']}, local_epochs: {PARAMS['local_epochs']}, full_dataset_size: {PARAMS['full_dataset_size']}, num_classes: {PARAMS['number_of_classes']}\n")
print(f"Privacy Params:\n epsilon: {PRIVACY_PARAMS['epsilon']}, target_epsilon: {PRIVACY_PARAMS['target_epsilon']}, target_delta: {PRIVACY_PARAMS['target_delta']}\n")
print(f"Device: {DEVICE}\n")


def save_str_to_file(string: str, dir_path: str) -> None:
    """Append a string to the log file in the specified directory."""
    with open(f"{dir_path}/log_file.txt", "a") as file:
        file.write(string + '\n')


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
    # epsilon = privacy_engine.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
    epsilon = PRIVACY_PARAMS["epsilon"]
    return epsilon, average_loss


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def test(net, testloader):
    """
    Evaluate the network on the test set.
    Returns (average_loss, accuracy, auc_score, precision, recall, f1_score, confusion_matrix).
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
    precision = precision_score(labels_array, predictions_array, average='weighted')
    recall = recall_score(labels_array, predictions_array, average='weighted')
    f1 = f1_score(labels_array, predictions_array, average='weighted')
    conf_matrix = confusion_matrix(labels_array, predictions_array)

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

    return normalized_fisher_diag


def update_noise_multiplier(
    base_noise_multiplier, fisher_diag, client_data_size, epoch, max_epochs, epsilon
):
    fisher_scalars = []
    for f in fisher_diag:
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()
        
        f_mean = np.mean(f)
        if(f_mean > 0.0001):
            fisher_scalars.append(np.mean(f))

    save_str_to_file("Fisher Scalars: " + str(fisher_scalars), client_name)

    noise_multipliers = [
        base_noise_multiplier * f  
        for f in fisher_scalars
    ]

    # Aggregate using mean
    noise_multiplier = np.mean(noise_multipliers)

    # convergence factor
    # if epoch > max_epochs // 2:
    #     convergence_factor = 1 - (epoch - max_epochs // 2) / (max_epochs // 2)
    #     noise_multiplier *= convergence_factor

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
        self.epsilon=PRIVACY_PARAMS["epsilon"]

        print("Step 1a: Client Initialized")


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
        state_dict = self.model.state_dict()
        new_state_dict = {}
        prefix = "_module."
        for key, value in state_dict.items():
            new_key = key[len(prefix):] if key.startswith(prefix) else key
            new_state_dict[new_key] = value
        clean_model.load_state_dict(new_state_dict)
        clean_model.to(self.device)
        
        print(f"Step 1b: Recomputing FIM for epoch {self.global_epoch}")
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
        self.epsilon, average_loss = train(
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
            f"Epsilon: {self.epsilon:.2f}, "
            f"Base Noise Multiplier: {base_noise_multiplier:.2f}, "
            f"Dynamic Noise Multiplier: {dynamic_noise_multiplier:.2f}"
        )
        save_str_to_file(log_str, client_name)

        self.global_epoch += 1

        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"epsilon": self.epsilon, "train_loss": average_loss}
        )


    def evaluate(self, parameters, config):
        """
        Evaluate the model locally using the given parameters,
        and return the loss, number of examples, and metrics.
        """
        print("Step 3: Evaluate the model locally")

        self.set_parameters(parameters)
        average_loss, accuracy, auc_score, precision, recall, f1, conf_matrix = test(self.model, self.testloader)

        actual_counts = conf_matrix.sum(axis=1)
        predicted_counts = conf_matrix.sum(axis=0)

        actual_counts_str = ", ".join([f"Class {i}: {cnt}" for i, cnt in enumerate(actual_counts)])
        predicted_counts_str = ", ".join([f"Class {i}: {cnt}" for i, cnt in enumerate(predicted_counts)])

        log_str = (
            f"Loss: {average_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"AUC: {auc_score:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1 Score: {f1:.4f},\n"
            f"Confusion Matrix:\n{conf_matrix}\n"
            f"Actual counts per class: {actual_counts_str}\n"
            f"Predicted counts per class: {predicted_counts_str}"
        )
        save_str_to_file(log_str, client_name)

        # Update client history
        client_history["loss"].append(average_loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc_score)
        client_history["precision"].append(precision)
        client_history["recall"].append(recall)
        client_history["f1"].append(f1)

        print(f"\n{client_history}\n")

        return float(average_loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
            "auc": float(auc_score),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }




if __name__ == "__main__":
    model = ViT_GPU(device=DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainload, testloader, sample_rate = load_data(client_index=0)

    init_log_str = f"Initial Train Dataset Size: {len(trainload.dataset)} Sample rate: {sample_rate}"
    save_str_to_file(init_log_str, client_name)

    fisher_diag = compute_fisher_information(model, trainload, device=device)
    client_data_size = len(trainload.dataset)

    fl.client.start_client(
        server_address="127.0.0.1:8052",
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
