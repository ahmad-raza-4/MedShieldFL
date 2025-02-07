import torch
import numpy as np

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
        data_scaling_factor = total_size / client_data_size
       
        # Convert tensors to numpy arrays for operations
        fisher_scaling_factor = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f for f in fisher_scaling_factor]

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

        noise_multiplier=np.mean(noise_multiplier)/epsilon
        
        print(f"Updated Noise Multiplier after division by epsilon {epsilon} for client size {client_data_size}: " , noise_multiplier)             

        return noise_multiplier