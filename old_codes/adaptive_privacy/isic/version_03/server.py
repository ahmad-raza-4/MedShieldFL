import os
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from time import time
from typing import List, Tuple, Optional, Dict, Any

from main import ViT  

CHECKPOINT_DIR = "/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/adaptive_privacy/isic/version_03/checkpoints"

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_checkpoint_round_")]
    if not checkpoints:
        return None, 0
    rounds = []
    for f in checkpoints:
        try:
            round_num = int(f.split('_')[-1].split('.')[0])
            rounds.append(round_num)
        except:
            continue
    if not rounds:
        return None, 0
    latest_round = max(rounds)
    latest_checkpoint = f"model_checkpoint_round_{latest_round}.pth"
    return os.path.join(checkpoint_dir, latest_checkpoint), latest_round

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, starting_round=1, **kwargs):
        super().__init__(**kwargs)
        self.starting_round = starting_round

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Any] 
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            model = ViT()
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            actual_round = self.starting_round + (server_round - 1)
            if actual_round % 5 == 0:
                checkpoint_path = f"{CHECKPOINT_DIR}/model_checkpoint_round_{actual_round}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[Server] Model checkpoint saved at round {actual_round} => {checkpoint_path}")

        return aggregated_parameters, aggregated_metrics

def main():
    start_time = time()

    # Load latest checkpoint
    checkpoint_path, latest_round = find_latest_checkpoint(CHECKPOINT_DIR)
    initial_parameters = None
    starting_round = 1
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model = ViT()
        model.load_state_dict(torch.load(checkpoint_path))
        ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
        initial_parameters = fl.common.ndarrays_to_parameters(ndarrays)
        starting_round = latest_round + 1
        print(f"Resuming training from round {starting_round}")
    else:
        print("No checkpoint found, starting from scratch")

    # Set total desired rounds and calculate remaining
    total_rounds = 30  # Adjust this to your desired total
    num_rounds = total_rounds - (starting_round - 1)
    if num_rounds <= 0:
        print("Training already completed. Adjust total_rounds if needed.")
        return

    strategy = SaveModelStrategy(
        starting_round=starting_round,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=6,
        min_evaluate_clients=6,
        min_available_clients=6,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8022",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    end_time = time()
    elapsed_hrs = (end_time - start_time) / 3600
    print(f"Time taken (hours): {elapsed_hrs}")

    try:
        if not os.path.exists("server"):
            os.makedirs("server")
        def save_str_to_file(string, dir: str):
            with open(f"{dir}/log_file.txt", "a") as file:
                file.write(string + "\n")

        save_str_to_file(f"Time taken (hours): {elapsed_hrs}", "server")
    except Exception as e:
        print(f"Failed to save time taken: {e}")

if __name__ == "__main__":
    main()