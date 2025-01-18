import torch
from flamby.datasets.fed_isic2019 import FedIsic2019
from torch.utils.data import DataLoader
from main import ViT_GPU 
import flwr as fl
from opacus import PrivacyEngine
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import numpy as np
from plot_graphs import plot_metrics
import os

# ---------- Additional Imports for Dynamic Noise Multiplier ----------
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import create_accountant
# ---------------------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
client_name = "client_4"

if not os.path.exists(client_name):
    os.makedirs(client_name)

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

PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "max_grad_norm": 1.0,
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_str_to_file(string, dir: str):
    with open(f"{dir}/log_file.txt", "a") as file:
        file.write(string + '\n')

def load_data(client_index: int):
    """
    Loads the training and test datasets for the specified client index.
    Returns trainloader, testloader, and sample_rate.
    """
    train_dataset = FedIsic2019(center=client_index, train=True)
    test_dataset = FedIsic2019(train=False)

    trainloader = DataLoader(train_dataset, batch_size=PARAMS["batch_size"])
    testloader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"])

    sample_rate = PARAMS["batch_size"] / len(train_dataset)
    return trainloader, testloader, sample_rate

def train(net, trainloader, privacy_engine, optimizer, epochs):
    """
    Standard training loop with Opacus privacy engine.
    Returns the privacy budget epsilon after training.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    # Retrieve the current epsilon for the chosen delta
    epsilon = privacy_engine.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
    return epsilon

def test(net, testloader):
    """
    Standard evaluation loop returning (loss, accuracy, AUC).
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    labels_list, scores_list = [], []
    correct, total_loss = 0, 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)

            # Compute softmax scores for AUC
            scores = torch.softmax(outputs, dim=1)
            labels_list.append(labels.cpu().numpy())
            scores_list.append(scores.cpu().numpy())

            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)

    # Compute AUC in multi-class setting
    labels_array = np.concatenate(labels_list)
    scores_array = np.concatenate(scores_list)
    auc_score = roc_auc_score(
        y_true=labels_array,
        y_score=scores_array,
        labels=list(range(8)),  # 8 classes in ISIC-2019, By providing the labels argument to roc_auc_score, scikit-learn knows that columns in scores_array correspond to classes [0..7], even if your local dataset lacks some labels. This typically resolves the shape mismatch if your model outputs have shape (N, 8).
        multi_class='ovr'
    )

    return total_loss, accuracy, auc_score

# ---------- Dynamic Noise Multiplier Function ----------
def compute_noise_multiplier4(
    target_epsilon,
    target_delta,
    global_epoch,
    local_epoch,
    batch_size,
    client_data_sizes,
    max_epochs,
    gradients=None,
    decay_factor=0.9
):
    """
    Provided function to compute an adjusted noise multiplier.
    Uses an accountant, base noise calculation, and optional gradient-based adjustments.
    """
    accountant = create_accountant(mechanism="prv")

    # Base noise based on target epsilon, delta, sampling rate, local epochs
    base_noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=1 / len(client_data_sizes),  # Simplified sampling rate assumption
        epochs=local_epoch,
        accountant=accountant.mechanism(),
    )

    avg_data_size = sum(client_data_sizes) / len(client_data_sizes)
    adjusted_noise_multipliers = []
    grad_scale_factor = 0.1

    for i, client_data_size in enumerate(client_data_sizes):
        # Basic scaling: bigger dataset => more noise
        client_noise_multiplier = base_noise_multiplier * (client_data_size / avg_data_size)

        # Optional gradient-based scaling, if needed
        if gradients is not None and i < len(gradients):
            grad_tensor = gradients[i]
            if isinstance(grad_tensor, list):
                grad_norms = [torch.norm(g).item() for g in grad_tensor]
                total_grad_norm = sum(grad_norms)
            else:
                total_grad_norm = torch.norm(grad_tensor).item()
            grad_based_noise_multiplier = grad_scale_factor / (total_grad_norm + 1e-6)
            client_noise_multiplier += grad_based_noise_multiplier

        adjusted_noise_multipliers.append(client_noise_multiplier)

    # Decay factor over global epochs (optional)
    if global_epoch > 1:
        adjusted_noise_multipliers = [
            nm * (decay_factor ** (global_epoch / max_epochs)) for nm in adjusted_noise_multipliers
        ]

    print(f"Epoch {global_epoch} - Adjusted noise multipliers: {adjusted_noise_multipliers}")

    string=f"Epoch {global_epoch} - Adjusted noise multipliers: {adjusted_noise_multipliers}"
    save_str_to_file(string, client_name)
    
    return adjusted_noise_multipliers
# --------------------------------------------------------

class FedViTDPClient4(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, sample_rate) -> None:
        """
        Client initialization with base references.
        Note: We'll do dynamic noise inside `fit()` for each global round.
        """
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.sample_rate = sample_rate

        # Store a global_epoch to track across rounds
        self.global_epoch = 1
        self.max_global_epochs = 30  # or match your total planned FL rounds

        # Create a base optimizer (will be re-wrapped in `fit`)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        # We'll initialize an empty PrivacyEngine here; it gets set properly in fit
        self.privacy_engine = None

    def get_parameters(self, config):
        """
        Get the locally updated parameters in NumPy format.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Set the model parameters from a list of NumPy arrays.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the model using the provided parameters, re-computing the noise multiplier each round.
        """
        # 1) Load incoming global model params
        self.set_parameters(parameters)

        # 2) Re-compute the noise multiplier (unchanged)
        dynamic_noises = compute_noise_multiplier4(
            target_epsilon=10.0,
            target_delta=PRIVACY_PARAMS["target_delta"],
            global_epoch=self.global_epoch,
            local_epoch=PARAMS["local_epochs"],
            batch_size=PARAMS["batch_size"],
            client_data_sizes=[len(self.trainloader.dataset)],
            max_epochs=self.max_global_epochs,
            gradients=None,
            decay_factor=0.9
        )
        dynamic_noise_multiplier = dynamic_noises[0]

        # 3) Re-initialize the PrivacyEngine with the newly computed noise
        self.privacy_engine = PrivacyEngine()

        # [CRUCIAL] Make sure the model is in train mode!
        self.model.train()

        # Re-create the optimizer so it can be wrapped again
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        # Opacus must see a model in train mode here:
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=dynamic_noise_multiplier,
        )

        # 4) Perform local DP training (unchanged)
        epsilon = train(
            self.model,
            self.trainloader,
            self.privacy_engine,
            self.optimizer,
            PARAMS["local_epochs"]
        )

        # Log training details
        string = f"Global Epoch (Round): {self.global_epoch}, Train Size: {len(self.trainloader.dataset)}, Sample Rate: {self.sample_rate}"
        save_str_to_file(string, client_name)
        print(f"Epsilon = {epsilon:.2f}")

        string = f"Epsilon = {epsilon:.2f}"
        save_str_to_file(string, client_name)

        # Increment global epoch for next round
        self.global_epoch += 1

        # Return updated parameters, number of training examples, and custom metrics
        return (
            self.get_parameters(config={}),
            len(self.trainloader),
            {"epsilon": epsilon}
        )

    def evaluate(self, parameters, config):
        """
        Evaluate the model using the provided parameters.
        """
        # 1) Load the global parameters
        self.set_parameters(parameters)

        # 2) Standard evaluation flow
        loss, accuracy, auc_score = test(self.model, self.testloader)
        client_history["loss"].append(loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc_score)

        # Log evaluation results
        string = f"Loss: {loss:.2f}, Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}"
        save_str_to_file(string, client_name)
        print(f"\n{client_history}\n")

        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

# ------------------- Main Execution -------------------
if __name__ == "__main__":
    model = ViT_GPU(device=DEVICE) 
    trainload, testloader, sample_rate = load_data(client_index=3)
    string = f"Initial Train Dataset Size: {len(trainload.dataset)} Sample rate: {sample_rate}"
    save_str_to_file(string, client_name)

    fl.client.start_client(
        server_address="127.0.0.1:8074",
        client=FedViTDPClient4(
            model=model,
            trainloader=trainload,
            testloader=testloader,
            sample_rate=sample_rate
        ).to_client()
    )

    plot_metrics(client_history, client_name)
    print(f"\n\nFinal client history:\n{client_history}\n")
