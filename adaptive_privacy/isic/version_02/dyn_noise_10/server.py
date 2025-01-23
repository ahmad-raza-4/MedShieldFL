import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from time import time
from typing import List, Tuple, Optional, Dict, Any

# Import the model architecture from your main.py
from main import ViT  # or ViT_GPU if needed

# Path where you want to save the model checkpoints
CHECKPOINT_DIR = "/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/adaptive_privacy/isic/version_02/dyn_noise_10/checkpoints"


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Overrides FedAvg to save the global model checkpoint after each round."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Any]  # Using Any for simplicity
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        # Call the aggregate_fit method of FedAvg to get aggregated parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` (Flower) to a list of NumPy ndarrays
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Load the aggregated parameters into our ViT model
            model = ViT()  # CPU model
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model checkpoint
            checkpoint_path = f"{CHECKPOINT_DIR}/model_checkpoint_round_{server_round}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Server] Model checkpoint saved at round {server_round} => {checkpoint_path}")

        return aggregated_parameters, aggregated_metrics


def main():
    start_time = time()

    # Instantiate our custom strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        # You can pass additional FedAvg constructor arguments if needed
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8072",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )

    end_time = time()
    elapsed_hrs = (end_time - start_time) / 3600
    print(f"Time taken (hours): {elapsed_hrs}")

    # Optional: Log how long the server ran
    try:
        def save_str_to_file(string, dir: str):
            with open(f"{dir}/log_file.txt", "a") as file:
                file.write(string + "\n")

        save_str_to_file(f"Time taken (hours): {elapsed_hrs}", "server")
    except Exception as e:
        print(f"Failed to save time taken: {e}")


if __name__ == "__main__":
    main()
