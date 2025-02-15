import torch
from torchvision import datasets, transforms
from main import ViT, ViT_GPU
import flwr as fl
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import numpy as np
from plot_graphs import plot_metrics
import os

torch.manual_seed(1)
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
client_name = "client_5"

if not os.path.exists(client_name):
    os.makedirs(client_name)

client_history = {
    "loss": [],
    "accuracy": [],
    "auc": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "confusion_matrix": []
}

PARAMS = {
    "batch_size": 32,
    "local_epochs": 3,
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_str_to_file(string, dir: str):
    with open(f"{dir}/log_file.txt", "a") as file:
        file.write(string + '\n')

def load_data(client_index: int):
    """Load training and testing data for the given client index."""
    # Paths to the training and testing directories
    train_dir = f"/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/train/client_{client_index}/"
    test_dir = "/home/dgxuser16/NTL/mccarthy/ahmad/github/data/alzheimers/test"

    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor() 
    ])

    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    testset = datasets.ImageFolder(root=test_dir, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=PARAMS["batch_size"], shuffle=True,
        num_workers=0, drop_last=True         
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=PARAMS["batch_size"], shuffle=False,
        num_workers=0
    )

    sample_rate = len(trainset) / 6400.0
    return trainloader, testloader, sample_rate

def train(net, trainloader, optimizer, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if images.size(0) == 0:
                continue  # Skip empty batches
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return None

# Add new imports at the top
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Update the test function
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    labels_list, scores_list, predicted_list = [], [], []
    correct, total_loss = 0, 0.0
    total_examples = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            scores = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            labels_list.append(labels.cpu().numpy())
            scores_list.append(scores.cpu().numpy())
            predicted_list.append(predicted.cpu().numpy())

            loss = criterion(outputs, labels).item()
            batch_size = images.size(0)
            total_loss += loss * batch_size
            total_examples += batch_size
            correct += (predicted == labels).sum().item()

    # Compute metrics
    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = correct / len(testloader.dataset)
    labels_array = np.concatenate(labels_list)
    predicted_array = np.concatenate(predicted_list)
    scores_array = np.concatenate(scores_list)

    # Calculate AUC
    auc_score = roc_auc_score(
        y_true=labels_array,
        y_score=scores_array,
        labels=list(range(4)), 
        multi_class='ovr'
    )

    # Calculate classification metrics
    cm = confusion_matrix(labels_array, predicted_array)
    precision = precision_score(labels_array, predicted_array, average='macro')
    recall = recall_score(labels_array, predicted_array, average='macro')
    f1 = f1_score(labels_array, predicted_array, average='macro')

    return average_loss, accuracy, auc_score, cm, precision, recall, f1


class FedViTClient5(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader) -> None:
        super().__init__()
        self.model = model
        self.testloader = testloader
        self.trainloader = trainloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        """Get the locally updated parameters in NumPy format."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model using the provided parameters and return the updated parameters."""
        self.set_parameters(parameters)
        train(
            self.model,
            self.trainloader,
            self.optimizer,
            PARAMS["local_epochs"]
        )
        string = f"Train Dataset Size: {len(self.trainloader.dataset)} Sample rate: {sample_rate}"
        save_str_to_file(string, client_name)
        
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {}  
        )

    # Modify the evaluate method in FedViTClient5
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, auc, cm, precision, recall, f1 = test(self.model, self.testloader)
        
        # Update history
        client_history["loss"].append(loss)
        client_history["accuracy"].append(accuracy)
        client_history["auc"].append(auc)
        client_history["precision"].append(precision)
        client_history["recall"].append(recall)
        client_history["f1"].append(f1)
        client_history["confusion_matrix"].append(cm)
        
        # Log metrics
        metrics_str = (
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n"
            f"Confusion Matrix:\n{np.array2string(cm)}"
        )
        save_str_to_file(metrics_str, client_name)
        
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy, "auc": auc}

model = ViT_GPU(device=DEVICE)

trainload, testloader, sample_rate = load_data(client_index=4)
string = f"Train Dataset Size: {len(trainload)} Sample rate: {sample_rate}"
save_str_to_file(string, client_name)

fl.client.start_client(
    server_address="127.0.0.1:8093",
    client=FedViTClient5(model=model, trainloader=trainload, testloader=testloader).to_client()
)

plot_metrics(client_history, client_name)
print(f"\n\n{client_history}")
