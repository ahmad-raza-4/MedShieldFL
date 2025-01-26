import flwr as fl
from time import time

start_time = time()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,  
    min_evaluate_clients=1,
    min_available_clients=1,
)

fl.server.start_server(
    server_address="0.0.0.0:8044",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
)

end_time = time()
print(f"Time taken (hours): {(end_time - start_time) / 3600}")


try:
    def save_str_to_file(string, dir: str):
        with open(f"{dir}/log_file.txt", "a") as file:
            file.write(string + '\n')

    string = f"Time taken (hours): {(end_time - start_time) / 3600}"
    save_str_to_file(string, "server")
except Exception as e:
    print(f"Failed to save time taken: {e}")