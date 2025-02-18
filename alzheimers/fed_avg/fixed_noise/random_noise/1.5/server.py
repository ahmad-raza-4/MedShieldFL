import flwr as fl
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

# strategy = fl.server.strategy.FedAvg()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=6,  
    min_evaluate_clients=6,
    min_available_clients=6,
)

# dp_strategy = DifferentialPrivacyClientSideFixedClipping(strategy=strategy, num_sampled_clients=6, noise_multiplier=1.0, clipping_norm=0.5)

fl.server.start_server(
    server_address="0.0.0.0:8719",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
)
