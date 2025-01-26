import copy
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader as dl
import torch.nn.functional as F

import sys
import importlib.util

from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

from flamby.utils import evaluate_model_on_tests
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

device = torch.device("cpu")

file_path = '/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/new_exp_2/flamby/strategies/fed_avg.py'

# Load the module dynamically
spec = importlib.util.spec_from_file_location("fed_avg", file_path)
strategies = importlib.util.module_from_spec(spec)
sys.modules["fed_avg"] = strategies
spec.loader.exec_module(strategies)

FedAvg = strategies.FedAvg

torch.multiprocessing.set_sharing_strategy("file_system")


def compute_fisher_information(model, dataloader, device):
    fisher_diag = [torch.zeros_like(param).to(device) for param in model.parameters()]
    model.eval()

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)

        outputs = torch.sigmoid(outputs).clamp(min=1e-6, max=1 - 1e-6)

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
    fisher_scaling_factor = [f + 1e-6 for f in fisher_diag]
    total_size = 740
    data_scaling_factor = total_size / avg_data_size

    fisher_scaling_factor = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in fisher_scaling_factor]

    for f in fisher_scaling_factor:
        noise_multiplier = base_noise_multiplier * f * data_scaling_factor

    noise_multiplier = noise_multiplier.tolist()

    if isinstance(noise_multiplier, list):
        noise_multiplier = sum(noise_multiplier) / len(noise_multiplier)

    if isinstance(noise_multiplier, torch.Tensor):
        noise_multiplier = noise_multiplier.item()

    noise_multiplier = noise_multiplier / epsilon
    return noise_multiplier


n_repetitions = 5
num_updates = 100
nrounds = get_nb_max_rounds(num_updates)

bloss = BaselineLoss()

args = {
    "loss": bloss,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": num_updates,
    "nrounds": nrounds,
}

epsilons = [0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0][::-1]
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

# Initialize the CSV file at the beginning
results_csv_path = "results_fed_heart_disease_u2.csv"
results_columns = ["perf", "epsilon", "delta", "seed"]
results_df = pd.DataFrame(columns=results_columns)
results_df.to_csv(results_csv_path, index=False)

results_all_reps = []
edelta_list = list(product(epsilons, deltas))
for se in seeds:
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

    for epsilon, delta in product(epsilons, deltas):
        accountant = create_accountant(mechanism="prv")
        base_noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=1 / 6,
            epochs=20,
            accountant=accountant.mechanism(),
        )
        total_size = 740
        avg_data_size = total_size / 6

        for client_idx, dl in enumerate(training_dls):
            fisher_diag = compute_fisher_information(Baseline(), dl, device=device)
            client_data_size = len(dl.dataset)
            dynamic_noise_multiplier = update_noise_multiplier(
                base_noise_multiplier, fisher_diag, client_data_size, avg_data_size, epsilon
            )
            edelta_list.append((epsilon, delta, dynamic_noise_multiplier))

    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)

    s = FedAvg(**current_args, log=False)
    cm = s.run()[0]
    mean_perf = np.array(
        [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
    ).mean()
    results_all_reps.append({"perf": mean_perf, "epsilon": None, "delta": None, "seed": se})

    # Write results after each iteration
    temp_results = pd.DataFrame(results_all_reps[-1:], columns=results_columns)
    temp_results.to_csv(results_csv_path, mode="a", header=False, index=False)

    for entry in tqdm(edelta_list):
        if len(entry) == 2:
            epsilon, delta = entry
            dynamic_noise_multiplier = None  # Set to None or skip this if unnecessary
        elif len(entry) == 3:
            epsilon, delta, dynamic_noise_multiplier = entry
        else:
            raise ValueError(f"Unexpected tuple length in edelta_list: {len(entry)}")
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["dp_target_epsilon"] = epsilon
        current_args["dp_target_delta"] = delta
        current_args["dp_max_grad_norm"] = 1.1

        s = FedAvg(**current_args, log=False)
        cm = s.run()[0]
        mean_perf = np.array(
            [v for _, v in evaluate_model_on_tests(cm, test_dls, metric).items()]
        ).mean()

        print(f"Mean performance eps={epsilon}, delta={delta}, Perf={mean_perf}, Seed={se}, Noise={dynamic_noise_multiplier}")

        results_all_reps.append({"perf": mean_perf, "epsilon": entry, "delta": delta, "seed": se})

        # Write results to the CSV file after each iteration
        temp_results = pd.DataFrame(results_all_reps[-1:], columns=results_columns)
        temp_results.to_csv(results_csv_path, mode="a", header=False, index=False)
