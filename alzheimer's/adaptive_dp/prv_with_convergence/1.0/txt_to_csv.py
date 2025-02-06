import csv
import re

# Input and output file names
input_file = txt_file_path = "/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/adaptive_privacy/alzheimer's/new/1.0/client_1/log_file.txt"  # replace with your .txt file path
output_file = "output.csv"

# Define a regex pattern to extract required values
pattern = re.compile(
    r"Global Epoch \(Round\): (\d+).*?"
    r"Loss: ([\d\.]+), Accuracy: ([\d\.]+), AUC: ([\d\.]+)", re.DOTALL
)

# Read the input file and extract relevant data
data = []
with open(input_file, "r") as file:
    content = file.read()
    matches = pattern.findall(content)
    for match in matches:
        epoch, loss, accuracy, auc = match
        data.append([int(epoch), float(auc), float(accuracy), float(loss)])

# Write data to CSV
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "auc", "acc", "loss"])
    writer.writerows(data)

print(f"CSV file '{output_file}' has been generated successfully.")