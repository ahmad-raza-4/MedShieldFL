"""
Federated Learning Client with Differential Privacy for Alzheimer's Disease Classification
"""

import os
import logging
import numpy as np
import torch
import flwr as fl
from opacus import PrivacyEngine
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import create_accountant
from torchvision import datasets, transforms
from main import ViT_GPU

# --------------------------
# Configuration Setup
# --------------------------

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration Constants
CLIENT_NAME = "client_1"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # Set GPU visibility

TRAIN_PARAMS = {
    "batch_size": 32,
    "local_epochs": 3,
    "full_dataset_size": 6400,
    "num_classes": 4
}

PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "max_grad_norm": 1.0,
    "target_epsilon": 30
}

# --------------------------
# Helper Functions
# --------------------------

def load_data(client_index: int) -> Tuple[DataLoader, DataLoader, float]:
    """Load partitioned training data and centralized test data.
    
    Args:
        client_index: Partition index for client-specific data
        
    Returns:
        Tuple containing trainloader, testloader, and sample rate
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor()
        ])

        # Path configuration
        train_dir = f"/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/train/client_{client_index}/"
        test_dir = "/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/test"


        # Dataset preparation
        trainset = datasets.ImageFolder(root=train_dir, transform=transform)
        testset = datasets.ImageFolder(root=test_dir, transform=transform)

        logger.info(f"Loaded datasets - Train: {len(trainset)} samples, Test: {len(testset)} samples")

        return (
            DataLoader(trainset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True),
            DataLoader(testset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=False),  # Fixed shuffle
            len(trainset) / TRAIN_PARAMS["full_dataset_size"]
        )
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def compute_fisher_information(model: torch.nn.Module, 
                              dataloader: DataLoader,
                              device: torch.device) -> List[torch.Tensor]:
    """Compute Fisher Information for noise adaptation.
    
    Args:
        model: Current model state
        dataloader: Training data loader
        device: Computation device
        
    Returns:
        List of Fisher information values per parameter
    """
    model.train()
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]

    try:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=1)

            for i, label in enumerate(labels):
                model.zero_grad()
                log_prob = log_probs[i, label]
                log_prob.backward(retain_graph=True)

                for j, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        fisher_diag[j] += param.grad.pow(2)
                
                model.zero_grad()

        return [fisher / len(dataloader.dataset) for fisher in fisher_diag]
    except RuntimeError as e:
        logger.error(f"Fisher computation failed: {str(e)}")
        raise

# --------------------------
# Core Client Class
# --------------------------

class FedViTDPClient(fl.client.NumPyClient):
    """Federated Client with Adaptive Differential Privacy"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 trainloader: DataLoader,
                 testloader: DataLoader,
                 sample_rate: float) -> None:
        super().__init__()
        
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.sample_rate = sample_rate
        self.privacy_engine = PrivacyEngine()
        self.global_round = 1

        # Initialize optimizer once
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=1e-4
        )

        logger.info(f"Client initialized with {len(trainloader.dataset)} training samples")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update model with received parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Federated training round with adaptive DP"""
        self.set_parameters(parameters)
        
        # Recompute Fisher information each round
        fisher_diag = compute_fisher_information(
            self.model, self.trainloader, DEVICE
        )
        
        # Calculate adaptive noise
        noise_multiplier = self._calculate_adaptive_noise(fisher_diag)
        
        # Configure privacy engine
        self._configure_privacy_engine(noise_multiplier)
        
        # Training execution
        epsilon, train_loss = self._train_model()
        
        # Logging and metrics
        self._log_training_metrics(epsilon, noise_multiplier, train_loss)
        
        self.global_round += 1
        
        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            {"epsilon": epsilon, "train_loss": train_loss}
        )

    def _calculate_adaptive_noise(self, fisher_diag: List[torch.Tensor]) -> float:
        """Calculate dynamic noise multiplier with Fisher scaling"""
        client_data_size = len(self.trainloader.dataset)
        
        # Base noise calculation
        base_noise = get_noise_multiplier(
            target_epsilon=PRIVACY_PARAMS["target_epsilon"],
            target_delta=PRIVACY_PARAMS["target_delta"],
            sample_rate=client_data_size / TRAIN_PARAMS["full_dataset_size"],
            epochs=TRAIN_PARAMS["local_epochs"],
            accountant=create_accountant(mechanism="prv").mechanism(),
        )
        
        # Fisher-adaptive scaling
        fisher_mean = torch.stack([f.mean() for f in fisher_diag]).mean().item()
        data_scale = client_data_size / TRAIN_PARAMS["full_dataset_size"]
        
        dynamic_noise = base_noise * fisher_mean * data_scale
        logger.info(f"Noise scaling - Base: {base_noise:.4f} → Dynamic: {dynamic_noise:.4f}")
        
        return dynamic_noise

    def _configure_privacy_engine(self, noise_multiplier: float) -> None:
        """Configure DP parameters for training"""
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=noise_multiplier
        )

    def _train_model(self) -> Tuple[float, float]:
        """Execute local training epoch"""
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, total_samples = 0.0, 0

        for epoch in range(TRAIN_PARAMS["local_epochs"]):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Log gradient statistics
                with torch.no_grad():
                    total_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)
                    
                    # Gradient monitoring
                    grads = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                    logger.debug(f"Gradient norms - Mean: {np.mean(grads):.4f} ± {np.std(grads):.4f}")

        return self.privacy_engine.get_epsilon(PRIVACY_PARAMS["target_delta"]), total_loss / total_samples

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local test set"""
        self.set_parameters(parameters)
        self.model.eval()
        
        criterion = torch.nn.CrossEntropyLoss()
        all_labels, all_scores = [], []
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                
                # Loss calculation
                total_loss += criterion(outputs, labels).item() * images.size(0)
                
                # Accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
                # AUC preparation
                all_labels.append(labels.cpu().numpy())
                all_scores.append(torch.softmax(outputs, dim=1).cpu().numpy())

        # Metrics calculation
        avg_loss = total_loss / len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        auc_score = roc_auc_score(
            np.concatenate(all_labels),
            np.concatenate(all_scores),
            multi_class='ovr'
        )

        logger.info(f"Evaluation - Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | AUC: {auc_score:.4f}")
        
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy}

    def _log_training_metrics(self, 
                            epsilon: float, 
                            noise: float, 
                            loss: float) -> None:
        """Standardized training metrics logging"""
        logger.info(
            f"Round {self.global_round} complete | "
            f"Epsilon: {epsilon:.2f} | "
            f"Noise Multiplier: {noise:.4f} | "
            f"Train Loss: {loss:.4f}"
        )

# --------------------------
# Execution Entry Point
# --------------------------

if __name__ == "__main__":
    try:
        # Model initialization
        model = ViT_GPU(device=DEVICE)
        logger.info("Model initialized on %s", DEVICE)
        
        # Data loading
        trainloader, testloader, sample_rate = load_data(client_index=0)
        
        # Start client
        fl.client.start_client(
            server_address="127.0.0.1:8057",
            client=FedViTDPClient(
                model=model,
                trainloader=trainloader,
                testloader=testloader,
                sample_rate=sample_rate
            ).to_client()
        )
        
    except Exception as e:
        logger.error("Client execution failed: %s", str(e), exc_info=True)
        raise