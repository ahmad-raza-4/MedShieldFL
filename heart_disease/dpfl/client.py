import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
from utils import Net, load_partition,get_data_loaders
from flamby_dataset import FedHeart
import argparse
from collections import OrderedDict
import logging, sys
import wandb
import warnings
from opacus import PrivacyEngine

# warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(DEVICE)
PRIVACY_PARAMS = {
    "target_delta": 1e-5,
    "noise_multiplier": 0.00411,
    "max_grad_norm": 1.0,
}

# nm=0.0270 for e=10

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
        train_loader, val_loader, _ = get_data_loaders(trainset, valset,0.2 ,batch_size=32)
        self.model=model
        self.trainloader=train_loader
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)
            
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
        """" Load Model here and replace its parameters when its given """

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """" Train model on the data (local data of each client) """
        print(f"[Client {self.cid}] fit, config: {config}")

        # Initialize wandb run
        '''
        run = wandb.init(project="FL4E-Experiments", config=config, group="Client " + str(self.cid),
                         job_type="client training")
        config = wandb.config
        '''

        # Update local model parameters
        self.model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = 32
        num_epochs: int =64
        lr = 0.001
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)
        train_dataset = self.trainset
        val_dataset = self.valset

        # Remove the outer loop over epochs
        print(f"Training for {num_epochs} epochs...")
        result = utils.train(model=self.model, train_dataset=train_dataset, val_dataset=val_dataset,
                             optimizer=self.optimizer, num_epochs=num_epochs,
                             batch_size=batch_size,
                             lr=lr)

        print(
            f" ROC_AUC: {result['val_roc_auc']:.4f}|| Accuracy {result['val_accuracy']:.4f} || Train Loss: {result['train_loss']:.4f}")
        print(
            f" Val Loss: {result['val_loss']:.4f} ")
        '''
        run.log({
            "epoch": num_epochs,
            "train_loss": result['train_loss'],
            "val_loss": result['val_loss'],
            "val_roc_auc": result['val_roc_auc'],
            "val_accuracy": result['val_accuracy'],
            "cid": self.cid
        })
        '''

        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(self.trainset)

        """returns the result of the last training epoch"""

        results = {"train_loss": result['train_loss'],
                   "train_roc_auc": result['val_roc_auc'],
                   "val_accuracy": result['val_accuracy'],
                   "val_loss": result['val_loss'],
                   "cid": self.cid}

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """" evaluate the model on locally hold test data """

        print(f"[Client {self.cid}] evaluate, config: {config}")

        # Update local model parameters
        self.model = self.set_parameters(parameters)

        # Get config for the evaluation
        batch_size: int = config["batch_size"]

        # Evaluate the global model on the local test data of each client and return results

        testset = self.testset
        loss, test_roc_auc, test_accuracy = utils.test(model=self.model, test_dataset=testset, batch_size=batch_size)
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
        '''
        wandb.log({
            "test_loss": metrics["loss"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_accuracy": metrics["test_accuracy"],
            "eval_cid": metrics["cid"]
        })
        '''
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

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # start client
    client = FL4EClients(args.cid, train_dataset, val_dataset, test_dataset, device)

    fl.client.start_numpy_client(server_address="0.0.0.0:8787", client=client)


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
