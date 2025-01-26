import copy
from itertools import product
import csv

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader as dl
from tqdm import tqdm

from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

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

# local imports
from noise import compute_fisher_information, update_noise_multiplier
from fed_avg import FedAvg as fed_avg
from flamby.utils import evaluate_model_on_tests

torch.multiprocessing.set_sharing_strategy("file_system")

csv_file = "heart_results.csv"
csv_headers = ["Iteration", "Epsilon", "Delta", "Noise", "Mean_Performance", "Seed"]

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)


n_repetitions = 1
num_updates = 100
nrounds = get_nb_max_rounds(num_updates)

FedAvg = fed_avg
bloss = BaselineLoss()
# We init the strategy parameters to the following default ones

args = {
    "loss": bloss,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": num_updates,
    "nrounds": nrounds,
}

epsilons = [1.0,10.0,20.0,30.0][::-1]
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
print(f"Total Combinations without Noise: {len(edelta_list)}")

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

    combinations=[]

    # Adaptive DP results
    for epsilon, delta in edelta_list:
        accountant = create_accountant(mechanism="prv")
        base_noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=0.25,  
            epochs=30,  
            accountant=accountant.mechanism(),
        )
        avg_data_size = 185
        
        for client_idx, train_dl in enumerate(training_dls):
            fisher_diag = compute_fisher_information(Baseline(), train_dl, device='cpu')
            client_data_size = len(train_dl.dataset)
            dynamic_noise_multiplier = update_noise_multiplier(
                base_noise_multiplier, fisher_diag, client_data_size, avg_data_size, epsilon
            )
            combinations.append((epsilon, delta, dynamic_noise_multiplier))
    
    print(f"Total Combinations with Noise: {len(combinations)}")

    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)
    args["dp_dynamic_noise_multiplier"] = dynamic_noise_multiplier


    for entry in tqdm(combinations):
        if len(entry) == 2:
            e, d = entry
            noise = None  
        elif len(entry) == 3:
            e, d, noise = entry
        else:
            raise ValueError(f"Unexpected tuple length in combinations: {len(entry)}")
        
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
        print(f"Mean performance eps={e}, delta={d}, Perf={mean_perf}, Noise={noise}")
        
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([len(results_all_reps), e, d, noise, mean_perf, se])

        results_all_reps.append({"perf": mean_perf, "e": e, "d": d, "seed": se})


results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv("results_fed_heart_disease_noise.csv", index=False)

