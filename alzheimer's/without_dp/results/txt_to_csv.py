import re
import csv

# File paths
input_file = "client_1/log_file.txt"  # Replace with the path to your .txt file
output_file = "client_1/log_file.csv"  # Replace with the desired output .csv file

# Regex patterns for extracting data
# train_dataset_pattern = r"Train Dataset Size: (\d+) Sample rate: ([\d.]+)"
metrics_pattern = r"Loss: ([\d.]+), Accuracy: ([\d.]+), AUC: ([\d.]+)"

# Lists to store extracted data
data = []

i=1
# Read the .txt file and extract data
with open(input_file, "r") as file:
    for line in file:
        # train_match = re.search(train_dataset_pattern, line)
        metrics_match = re.search(metrics_pattern, line)
        
        # if train_match:
        #     train_dataset_size = train_match.group(1)
        #     sample_rate = train_match.group(2)
        
        if metrics_match:
            loss = metrics_match.group(1)
            accuracy = metrics_match.group(2)
            auc = metrics_match.group(3)
            # Append the extracted data to the list
            data.append([i, loss, accuracy, auc])
            i+=1

# Write the data to a .csv file
with open(output_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header row
    csv_writer.writerow(["Epoch", "Loss", "Accuracy", "AUC"])
    # Write the data rows
    csv_writer.writerows(data)

print(f"Data successfully written to {output_file}.")
