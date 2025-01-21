import flwr as fl
import torch
from time import time
from collections import OrderedDict

# Import your model definition (e.g., ViT_GPU) from your "main.py" or wherever it's defined
from main import ViT_GPU

def set_parameters_torch(model: torch.nn.Module, parameters: list) -> None:
    """
    Load parameters (numpy.ndarray) into a PyTorch model's state_dict.
    The ordering of 'parameters' must match the model's .state_dict().
    """
    # Ensure keys match between model state_dict and incoming parameters
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    Custom strategy that stores the aggregated parameters each round 
    and saves the model at the end of training using on_conclude.
    """
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        # Keep a reference to a model on the server (usually on CPU)
        self.model = model
        self.final_round_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        # Call super to do the standard FedAvg aggregation
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)

        # If aggregation succeeded, store the aggregated_parameters 
        if aggregated_parameters is not None:
            self.final_round_parameters = aggregated_parameters

        return aggregated_parameters

    def on_conclude(self) -> None:
        """
        Called by Flower once the last round (num_rounds) has finished.
        We can load the final aggregated parameters into self.model
        and save it to disk.
        """
        if self.final_round_parameters is not None:
            # Load final aggregated parameters into our model
            set_parameters_torch(self.model, self.final_round_parameters)
            # Save the model state_dict to a file
            torch.save(self.model.state_dict(), "final_model.pth")
            print("Saved final global model to 'final_model.pth'")
        else:
            print("No final parameters found. Model not saved.")

def save_str_to_file(string: str, dir: str):
    """
    Example helper to log times or other info to a file in `dir`.
    """
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(f"{dir}/log_file.txt", "a") as file:
        file.write(string + '\n')

if __name__ == "__main__":
    start_time = time()

    # Instantiate a model on the server side (on CPU by default)
    global_model = ViT_GPU(device="cpu")
    
    # Create an instance of our custom strategy
    strategy = SaveModelStrategy(
        model=global_model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=6,
        min_evaluate_clients=6,
        min_available_clients=6,
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8084",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )

    end_time = time()
    hours_taken = (end_time - start_time) / 3600
    print(f"Time taken (hours): {hours_taken}")

    # Optionally save time to a log file
    try:
        string = f"Time taken (hours): {hours_taken}"
        save_str_to_file(string, "server")
    except Exception as e:
        print(f"Failed to save time taken: {e}")
