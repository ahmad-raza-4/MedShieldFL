# without saving of model weights

import flwr as fl
import torch
from main import ViT_GPU

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
    eta  = 1e-3,
    eta_l  = 1e-3,
    tau  = 1e-9, 
)


fl.server.start_server(
    server_address="0.0.0.0:8079",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
)


# import flwr as fl
# import torch
# from collections import OrderedDict
# from main import ViT_GPU
# import os


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ViT_GPU(device=device).to(device)
# model.eval()


# save_dir = "/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/isic/adagrad/e=10/models"
# os.makedirs(save_dir, exist_ok=True)


# def get_parameters(model):
#     """Extract and return model parameters as a list of NumPy arrays."""
#     return [val.cpu().numpy() for _, val in model.state_dict().items()]


# def save_model(model, epoch):
#     """Save the model to a .pth file after every 10 epochs."""
#     if epoch % 10 == 0:
#         model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
#         torch.save(model.state_dict(), model_path)
#         print(f"Model saved at epoch {epoch} to {model_path}")

# initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model))


# strategy = fl.server.strategy.FedAdagrad(
#     fraction_fit=1.0,
#     fraction_evaluate=1.0,
#     min_fit_clients=6,
#     min_evaluate_clients=6,
#     min_available_clients=6,
#     initial_parameters=initial_parameters,
#     eta  = 1e-2,
#     eta_l  = 1e-2,
#     tau  = 1e-9, 
# )


# class CustomStrategy(fl.server.strategy.FedAdagrad):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


#     def aggregate_fit(self, rnd, results, failures):
#         """Override to save model after every 10 epochs."""
#         model = super().aggregate_fit(rnd, results, failures)
#         save_model(model, rnd)
#         return model


# strategy = CustomStrategy(
#     fraction_fit=1.0,
#     fraction_evaluate=1.0,
#     min_fit_clients=6,
#     min_evaluate_clients=6,
#     min_available_clients=6,
#     initial_parameters=initial_parameters 
# )


# fl.server.start_server(
#     server_address="0.0.0.0:8457",
#     config=fl.server.ServerConfig(num_rounds=30),
#     strategy=strategy,
# )



