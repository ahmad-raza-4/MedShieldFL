import flwr as fl

from flwr.server.strategy import DifferentialPrivacyClientSideAdaptiveClipping 

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=6,  
    min_evaluate_clients=6,
    min_available_clients=6,
)

dp_strategy = DifferentialPrivacyClientSideAdaptiveClipping(
    strategy=strategy,
    noise_multiplier=0.8,
    num_sampled_clients=6,
    clipped_count_stddev=0.5
)

fl.server.start_server(
    server_address="0.0.0.0:8067",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=dp_strategy,
)
