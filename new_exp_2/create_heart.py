import copy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

import sys
import importlib.util

from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
import torch.nn.functional as F

device = torch.device("cpu")

# Define the absolute path to 'strategies.py'
file_path = '/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/new_exp_2/flamby/strategies/fed_avg.py'

# Load the module dynamically
spec = importlib.util.spec_from_file_location("fed_avg", file_path)
strategies = importlib.util.module_from_spec(spec)
sys.modules["fed_avg"] = strategies
spec.loader.exec_module(strategies)

# Now you can access FedAvg from the strategies module
FedAvg = strategies.FedAvg

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    get_nb_max_rounds,
    metric,
)
#from flamby.strategies import FedAvg
from flamby.utils import evaluate_model_on_tests

torch.multiprocessing.set_sharing_strategy("file_system")

# def compute_fisher_information(model, dataloader, device):
#     """
#     Compute Fisher Information (diagonal approximation) for each parameter of the model.
#     """
#     fisher_diag = [torch.zeros_like(param).to(device) for param in model.parameters()]
#     model.eval()

#     for data, labels in dataloader:
#         data, labels = data.to(device), labels.to(device)
#         outputs = model(data)
#         outputs = model(data)
#         print(f"Outputs: {outputs}")
#         print(f"Outputs shape: {outputs.shape}")

#         log_probs = F.log_softmax(outputs, dim=1)

#         for i, label in enumerate(labels):
#             print(f"Processing label {label}")
#             print(f"label size {label.size()}")
#             print(f"label item {label.item()}")

#             log_prob = log_probs[i, int(label.item())]  # Convert label to integer
#             model.zero_grad()
#             log_prob.backward(retain_graph=True)

#             for j, param in enumerate(model.parameters()):
#                 if param.grad is not None:
#                     fisher_diag[j] += param.grad ** 2
#             model.zero_grad()


#     fisher_diag = [fisher_value / len(dataloader.dataset) for fisher_value in fisher_diag]
#     return fisher_diag

def compute_fisher_information(model, dataloader, device):
    """
    Compute Fisher Information (diagonal approximation) for each parameter of the model.
    """
    fisher_diag = [torch.zeros_like(param).to(device) for param in model.parameters()]
    model.eval()

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)

        # Ensure outputs are probabilities between 0 and 1
        outputs = torch.sigmoid(outputs).clamp(min=1e-6, max=1 - 1e-6)

        # Compute log probabilities for binary classification
        log_probs = torch.log(outputs) * labels + torch.log(1 - outputs) * (1 - labels)

        for i, log_prob in enumerate(log_probs):
            model.zero_grad()
            log_prob.backward(retain_graph=True)

            for j, param in enumerate(model.parameters()):
                if param.grad is not None:
                    fisher_diag[j] += param.grad ** 2
            model.zero_grad()

    fisher_diag = [fisher_value / len(dataloader.dataset) for fisher_value in fisher_diag]
    return fisher_diag


def update_noise_multiplier(base_noise_multiplier, fisher_diag, client_data_size, avg_data_size,epsilon):
        """
        Update the noise multiplier dynamically based on base noise, Fisher information, and client data scale.
        """
        # Fisher scaling: parameters with higher Fisher information should have lower noise
        #fisher_scaling_factor = ([1.0 / (f + 1e-6) for f in fisher_diag])  # Avoid division by zero
        fisher_scaling_factor = [f + 1e-6 for f in fisher_diag]
        print("BaseNM",base_noise_multiplier)


        # Client data scale: clients with more data need less noise
        total_size=24459
        data_scaling_factor = total_size / avg_data_size
       
        # Ensure that each element is a scalar or a float value
        fisher_scaling_factor = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in fisher_scaling_factor]

        # Iterate over each tensor in fisher_scaling_factor and apply operations individually
        for f in fisher_scaling_factor:
           noise_multiplier = base_noise_multiplier * f * data_scaling_factor
      

        # Combine base noise multiplier with Fisher scaling and data scaling
        #self.noise_multiplier = base_noise_multiplier * fisher_scaling_factor * data_scaling_factor
        noise_multiplier = noise_multiplier.tolist()  # Convert to list for each parameter
        
        # Ensure it's a scalar float
        if isinstance(noise_multiplier, list):
            noise_multiplier = sum(noise_multiplier) / len(noise_multiplier)  # Example: average if list
        
        if isinstance(noise_multiplier, torch.Tensor):
            noise_multiplier = noise_multiplier.item()  # Convert tensor to float if needed
        
        print("noise multiplier" , noise_multiplier)
        noise_multiplier=noise_multiplier/epsilon
        
        
        print(f"Noise multiplier before  adjustment: {noise_multiplier}")
             

        return noise_multiplier
        

n_repetitions = 5
num_updates = 100
nrounds = get_nb_max_rounds(num_updates)


bloss = BaselineLoss()
# We init the strategy parameters to the following default ones

args = {
    "loss": bloss,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": num_updates,
    "nrounds": nrounds,
}

epsilons = [0.1, 1.0, 5.0, 10.0, 20.0,30.0,50.0][::-1]
deltas = [10 ** (-i) for i in range(1, 5)]
START_SEED = 42
seeds = np.arange(START_SEED, START_SEED + n_repetitions).tolist()

test_dls = [
    dl(
        FedHeartDisease(center=i, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        # collate_fn=collate_fn,
    )
    for i in range(NUM_CLIENTS)
]

results_all_reps = []
edelta_list = list(product(epsilons, deltas))
for se in seeds:
    # We set model and dataloaders to be the same for each rep
    global_init = Baseline()
    torch.manual_seed(se)
    training_dls = [
        dl(
            FedHeartDisease(center=i, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            # collate_fn=collate_fn,
        )
        for i in range(NUM_CLIENTS)
    ]
    args["training_dataloaders"] = training_dls
    # Loop through the combinations of epsilons and deltas
    for epsilon, delta in product(epsilons, deltas):
        # Create the accountant (assumes it uses some mechanism like "prv")
        accountant = create_accountant(mechanism="prv")
        # Compute the base noise multiplier (not used directly here)
        base_noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=1 / 6,  # Replace with your actual sample rate
        epochs=20,  # Replace with your actual local epochs
        accountant=accountant.mechanism(),)
        total_size = 23250
        avg_data_size = total_size / 6
        
        # Iterate over each client and their corresponding data loader
        for client_idx, dl in enumerate(training_dls):
            # Compute Fisher information for the current client (client-specific DataLoader)
            fisher_diag = compute_fisher_information(Baseline(), dl, device=device)  # Pass individual DataLoader
            # Get the size of the client's data
            client_data_size = len(dl.dataset)
            # Update the noise multiplier dynamically based on various factors
            dynamic_noise_multiplier = update_noise_multiplier(base_noise_multiplier,fisher_diag,  client_data_size,  avg_data_size,epsilon)
            # Append the result as a tuple (epsilon, delta, dynamic_noise_multiplier)
            edelta_list.append((epsilon, delta, dynamic_noise_multiplier))
        # If you need to view the results
        for entry in edelta_list:
            print(entry)

    
    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)

    # We run FedAvg wo DP
    s = FedAvg(**current_args, log=False)
    cm = s.run()[0]
    mean_perf = np.array(
        [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
    ).mean()
    print(f"Mean performance without DP, Perf={mean_perf}")
    results_all_reps.append({"perf": mean_perf, "e": None, "d": None, "seed": se})
    # args["dp_dynamic_noise_multiplier"] = dynamic_noise_multiplier

    for e, d in tqdm(edelta_list):
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["dp_target_epsilon"] = e
        current_args["dp_target_delta"] = d
        current_args["dp_max_grad_norm"] = 1.1
        # We run FedAvg
        s = FedAvg(**current_args, log=False)
        cm = s.run()[0]
        mean_perf = np.array(
            [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
        ).mean()
        print(f"Mean performance eps={e}, delta={d}, Perf={mean_perf}")
        # mean_perf = float(np.random.uniform(0, 1.))
        results_all_reps.append({"perf": mean_perf, "e": e, "d": d, "seed": se})

results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv("results_fed_heart_disease.csv", index=False)
