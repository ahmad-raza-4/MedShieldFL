# Standard library imports
import copy
from itertools import product
import warnings

warnings.filterwarnings("ignore")

# Third-party library imports
import csv
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader as dl
import torch.nn.functional as F
from tqdm import tqdm
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

# Local imports
from fed_avg import FedAvg as fed_avg
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
from flamby.utils import evaluate_model_on_tests

# Set multiprocessing strategy
torch.multiprocessing.set_sharing_strategy("file_system")

# Create a CSV file to store the results
csv_file = "heart_results.csv"
csv_headers = ["Iteration", "Epsilon", "Delta", "Dynamic_Noise_Multiplier", "Mean_Performance", "Seed"]

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)


# Global variables
FedAvg = fed_avg
bloss = BaselineLoss()

n_repetitions = 5
num_updates = 100
nrounds = get_nb_max_rounds(num_updates)

device = torch.device("cpu")


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



def update_noise_multiplier(base_noise_multiplier, fisher_diag, client_data_size, avg_data_size, epsilon):
        """
        Update the noise multiplier dynamically based on base noise, Fisher information, and client data scale.
        """
        # Fisher scaling factor: clients with more complex data need more noise
        fisher_scaling_factor = [f + 1e-6 for f in fisher_diag]
        print("Base Noise Multiplier: ",base_noise_multiplier)


        total_size=740
        data_scaling_factor = total_size / avg_data_size
       
        # Convert tensors to numpy arrays for operations
        fisher_scaling_factor = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in fisher_scaling_factor]

        # # Compute noise multiplier for each parameter
        # for f in fisher_scaling_factor:
        #    noise_multiplier = base_noise_multiplier * f * data_scaling_factor
      
        # # Convert to list for each parameter
        # noise_multiplier = noise_multiplier.tolist()

        # # Compute the average noise multiplier
        # if isinstance(noise_multiplier, list):
        #     noise_multiplier = sum(noise_multiplier) / len(noise_multiplier)
        
        # if isinstance(noise_multiplier, torch.Tensor):
        #     noise_multiplier = noise_multiplier.item()


        # Compute noise multipliers for all parameters
        noise_multipliers = [
            base_noise_multiplier * f * data_scaling_factor for f in fisher_scaling_factor
        ]

        # Convert each multiplier to list or retain as-is
        noise_multipliers = [
            n.tolist() if isinstance(n, torch.Tensor) else n for n in noise_multipliers
        ]

        # Compute the average noise multiplier
        if isinstance(noise_multipliers, list):
            noise_multiplier = sum(noise_multipliers) / len(noise_multipliers)
        elif isinstance(noise_multipliers, torch.Tensor):
            noise_multiplier = noise_multipliers.mean().item()

        
        # print("Updated Noise Multiplier: " , noise_multiplier)

        noise_multiplier=np.mean(noise_multiplier)/epsilon
        
        print(f"Updated Noise Multiplier after division by epsilon {epsilon}: " , noise_multiplier)             

        return noise_multiplier
        

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
    )
    for i in range(NUM_CLIENTS)
]

results_all_reps = []
edelta_list = list(product(epsilons, deltas))


for se in seeds:
    # Set the seed
    global_init = Baseline()
    torch.manual_seed(se)
    
    training_dls = [
        dl(
            FedHeartDisease(center=i, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        for i in range(NUM_CLIENTS)
    ]
    
    args["training_dataloaders"] = training_dls
    
    # Loop through the combinations of epsilons and deltas
    for epsilon, delta in product(epsilons, deltas):
        accountant = create_accountant(mechanism="prv")
        base_noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=1 / 4,  
            epochs=30,  
            accountant=accountant.mechanism(),
        )
        total_size = 740
        avg_data_size = total_size / 4
        
        for client_idx, train_dl in enumerate(training_dls):
            fisher_diag = compute_fisher_information(Baseline(), train_dl, device=device)
            client_data_size = len(train_dl.dataset)
            dynamic_noise_multiplier = update_noise_multiplier(
                base_noise_multiplier, fisher_diag, client_data_size, avg_data_size, epsilon
            )
            edelta_list.append((epsilon, delta, dynamic_noise_multiplier))
            
        # for entry in edelta_list:
        #     print(entry)
    
    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)

    # Run FedAvg without DP
    s = FedAvg(**current_args, log=False)
    cm = s.run()[0]
    mean_perf = np.array(
        [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
    ).mean()
    print(f"Mean performance without DP, Perf={mean_perf}")
    results_all_reps.append({"perf": mean_perf, "e": None, "d": None, "seed": se})
    args["dp_dynamic_noise_multiplier"] = dynamic_noise_multiplier

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([len(results_all_reps), None, None, dynamic_noise_multiplier, mean_perf, se])

    for entry in tqdm(edelta_list):
        if len(entry) == 2:
            e, d = entry
            noise = None  
        elif len(entry) == 3:
            e, d, noise = entry
        else:
            raise ValueError(f"Unexpected tuple length in edelta_list: {len(entry)}")
        
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["dp_target_epsilon"] = e
        current_args["dp_target_delta"] = d
        current_args["dp_max_grad_norm"] = 1.1
        
        # Run FedAvg
        s = FedAvg(**current_args, log=False)
        cm = s.run()[0]
        mean_perf = np.array(
            [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
        ).mean()
        print(f"Mean performance eps={e}, delta={d}, Perf={mean_perf}")
        
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([len(results_all_reps), e, d, dynamic_noise_multiplier, mean_perf, se])

        results_all_reps.append({"perf": mean_perf, "e": e, "d": d, "seed": se})

results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv("results_fed_heart_disease.csv", index=False)
