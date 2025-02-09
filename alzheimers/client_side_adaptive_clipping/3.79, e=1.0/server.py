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

strategy = DifferentialPrivacyClientSideFixedClipping(
    strategy,
    noise_multiplier=3.79,
    clipping_norm=1.0,
    num_sampled_clients=6,
)

fl.server.start_server(
    server_address="0.0.0.0:8034",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
)
