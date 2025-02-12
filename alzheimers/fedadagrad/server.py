import flwr as fl
import torch
from collections import OrderedDict
from main import ViT_GPU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT_GPU(device=device).to(device)
model.eval()


def get_parameters(model):
    """Extract and return model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model))


strategy = fl.server.strategy.FedAdagrad(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=6,
    min_evaluate_clients=6,
    min_available_clients=6,
    initial_parameters=initial_parameters,
    eta  = 1e-2,
    eta_l  = 1e-2,
    tau  = 1e-9, 
)


fl.server.start_server(
    server_address="0.0.0.0:8051",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
)
