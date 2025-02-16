import re
import csv

# Input and output file names
input_file = "e30.txt"  # Change to your actual file path
output_file = "e30.csv"

# Regex pattern to extract the required values
pattern = r"Global Epoch \(Round\): (\d+), .*Base Noise Multiplier: ([\d.]+), Dynamic Noise Multiplier: ([\d.]+)"

# List to store extracted data
data = []

# Read the input file
with open(input_file, "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            base_nm = float(match.group(2))
            dynamic_nm = float(match.group(3))
            data.append([epoch, base_nm, dynamic_nm])

# Write to CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Epoch", "Base NM", "Dynamic NM"])  # Header row
    writer.writerows(data)

print(f"Extraction complete. Data saved in '{output_file}'")
