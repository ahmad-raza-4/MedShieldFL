import re
import csv

# Define the input and output file paths
input_file = 'log_file.txt'
output_file = 'output.csv'

# Read the contents of the input file
with open(input_file, 'r') as f:
    content = f.read()

# Define a regex pattern to capture the epoch, loss, accuracy, and AUC.
# The pattern uses re.DOTALL to allow matching across multiple lines.
pattern = re.compile(
    r"Global Epoch \(Round\):\s*(\d+).*?"
    r"Loss:\s*([\d.]+),\s*Accuracy:\s*([\d.]+),\s*AUC:\s*([\d.]+)",
    re.DOTALL
)

# Find all matches in the file
matches = pattern.findall(content)

# Write the extracted data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["epoch", "auc", "acc", "loss"])
    
    # Write each row in the desired order: epoch, auc, acc, loss
    for match in matches:
        epoch, loss, acc, auc = match
        writer.writerow([epoch, auc, acc, loss])

print(f"Data extraction complete. CSV saved as '{output_file}'.")
