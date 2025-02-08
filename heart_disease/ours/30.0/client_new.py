import utils
import torchvision.datasets
import torch
import flwr as fl
from utils import Net, load_partition, get_data_loaders
from flamby_dataset import FedHeart
import argparse
from collections import OrderedDict
import logging, sys
import wandb
from opacus import PrivacyEngine
import numpy as np
import torch.nn.functional as F
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import create_accountant

# warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net().to(DEVICE)

PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "noise_multiplier": 0,
    "max_grad_norm": 1.0,
    "target_epsilon": 30.0,
    "epsilon": 30.0,
}

def compute_fisher_information(model, dataloader, device):
    """
    Compute the Fisher Information (diagonal approximation) for each parameter,
    adapted for a binary classification model with a single output.
    """
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]
    model.train()

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        labels = labels.squeeze().float()
        
        outputs = model(data)  # shape: (batch_size, 1)
     
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        losses = loss_fn(outputs.squeeze(), labels)
        
        # Compute gradients for each sample separately
        for i in range(len(labels)):
            model.zero_grad()
            losses[i].backward(retain_graph=True)
            with torch.no_grad():
                for j, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        fisher_diag[j] += param.grad ** 2

    # Normalize by the total dataset size
    fisher_diag = [fisher_value / len(dataloader.dataset) for fisher_value in fisher_diag]

    # Normalize Fisher values layer-wise
    normalized_fisher_diag = []
    for fisher_value in fisher_diag:
        x_min = torch.min(fisher_value)
        x_max = torch.max(fisher_value)
        normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min + 1e-8)
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
        if f_mean > 0.0001:
            fisher_scalars.append(np.mean(f))

    noise_multipliers = [
        base_noise_multiplier * f  
        for f in fisher_scalars
    ]

    noise_multiplier = 0.0
    if noise_multipliers == []:
        noise_multiplier = base_noise_multiplier
    else:
        # Aggregate using mean
        noise_multiplier = np.mean(noise_multipliers)

    # Convergence factor
    if epoch > max_epochs // 2:
        convergence_factor = 1 - (epoch - max_epochs // 2) / (max_epochs // 2)
        noise_multiplier *= convergence_factor

    return float(noise_multiplier)


class FL4EClients(fl.client.NumPyClient):
    def __init__(
            self,
            cid,
            trainset: torchvision.datasets,
            valset: torchvision.datasets,
            testset: torchvision.datasets,
            device: str,
    ):
        self.cid = cid
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.device = device
        self.privacy_engine = PrivacyEngine()
        train_loader, val_loader, _ = get_data_loaders(trainset, valset, 0.2, batch_size=32)
        self.model = model
        self.trainloader = train_loader
        self.fisher_diag = compute_fisher_information(model, train_loader, device)
        self.global_epoch = 1
        self.max_global_epochs = 30

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)

        # Initialize PrivacyEngine with base noise multiplier
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=PRIVACY_PARAMS["noise_multiplier"],
        )

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load Model here and replace its parameters when given."""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")

        self.model = self.set_parameters(parameters)

        model_copy = Net().to(self.device)
        model_copy.load_state_dict(self.model.state_dict())
        model_copy.train()
        self.fisher_diag = compute_fisher_information(model_copy, self.trainloader, self.device)

        client_data_size = len(self.trainloader.dataset)
        accountant = create_accountant(mechanism="prv")
        base_noise_multiplier = get_noise_multiplier(
            target_epsilon=PRIVACY_PARAMS["target_epsilon"],
            target_delta=PRIVACY_PARAMS["target_delta"],
            sample_rate=client_data_size / len(self.trainset),
            epochs=3,
            accountant=accountant.mechanism(),
        )
        print("base noise multiplier: ", base_noise_multiplier)
        
        dynamic_noise_multiplier = update_noise_multiplier(
            base_noise_multiplier=base_noise_multiplier,
            fisher_diag=self.fisher_diag,
            client_data_size=client_data_size,
            epoch=self.global_epoch,
            max_epochs=self.max_global_epochs,
            epsilon=PRIVACY_PARAMS["epsilon"],
        )

        print(f"Dynamic noise multiplier: {dynamic_noise_multiplier}")

        # --- Reinitialize PrivacyEngine with the dynamic noise multiplier ---
        # Create a fresh copy of the model to avoid wrapping hooks twice.
        model_clean = Net().to(self.device)
        model_clean.load_state_dict(self.model.state_dict())

        optimizer = torch.optim.Adam(model_clean.parameters(), lr=0.001, weight_decay=0.05)

        train_loader, _, _ = get_data_loaders(self.trainset, self.valset, 0.2, batch_size=32)

        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=model_clean,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=dynamic_noise_multiplier,
        )

        batch_size: int = 32
        num_epochs: int = config.get("local_epochs", 64)
        lr = 0.001

        result = utils.train(
            model=self.model,
            train_dataset=self.trainset,
            val_dataset=self.valset,
            optimizer=self.optimizer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
        )
        
        print(f" ROC_AUC: {result['val_roc_auc']:.4f}|| Accuracy {result['val_accuracy']:.4f} || Train Loss: {result['train_loss']:.4f}")
        print(f" Val Loss: {result['val_loss']:.4f} ")

        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(self.trainset)
        
        results = {
            "train_loss": result['train_loss'],
            "train_roc_auc": result['val_roc_auc'],
            "val_accuracy": result['val_accuracy'],
            "val_loss": result['val_loss'],
            "cid": self.cid,
            "dynamic_noise_multiplier": dynamic_noise_multiplier,
        }

        self.global_epoch += 1

        return parameters_prime, num_examples_train, results



    def evaluate(self, parameters, config):
        """Evaluate the model on locally held test data."""
        print(f"[Client {self.cid}] evaluate, config: {config}")

        self.model = self.set_parameters(parameters)

        batch_size: int = config["batch_size"]

        loss, test_roc_auc, test_accuracy = utils.test(
            model=self.model, test_dataset=self.testset, batch_size=batch_size
        )
        num_examples = len(self.testset)

        metrics = {
            "loss": float(loss),
            "test_roc_auc": float(test_roc_auc),
            "test_accuracy": float(test_accuracy),
            "cid": self.cid,
        }
        print(f"test_loss: {metrics['loss']}")
        print(f"test_roc_auc: {metrics['test_roc_auc']}")
        print(f"test_accuracy: {metrics['test_accuracy']}")
        print(f"eval_cid: {metrics['cid']}")

        return float(loss), num_examples, metrics


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="FL4E Fully Federated")
    parser.add_argument(
        "--cid",
        type=int,
        default=0,
        choices=range(0, 4),
        required=True,
        help="Specifies the CID (Client ID)",
    )
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_partition(args.cid)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start client
    client = FL4EClients(args.cid, train_dataset, val_dataset, test_dataset, device)

    fl.client.start_numpy_client(server_address="0.0.0.0:8757", client=client)


if __name__ == "__main__":
    import resource
    import time

    # Start measuring time
    start_time = time.time()
    # Start measuring resource usage
    usage_start = resource.getrusage(resource.RUSAGE_SELF)

    # Your script execution here
    main()
    wandb.finish()

    # End measuring resource usage
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    # End measuring time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    # Calculate CPU and RAM usage
    cpu_time = usage_end.ru_utime - usage_start.ru_utime
    ram_usage = (usage_end.ru_maxrss - usage_start.ru_maxrss) / (1024 * 1024)  # Convert to megabytes

    print(f"CPU Time: {cpu_time} seconds")
    print(f"Elapsed Time: {elapsed_time} seconds")
    print(f"RAM Usage: {ram_usage} megabytes")

    print('Logs saved in current directory')

    # After your script finishes executing, close the log file and restore stdout
    logging.shutdown()
    sys.stdout = sys.__stdout__