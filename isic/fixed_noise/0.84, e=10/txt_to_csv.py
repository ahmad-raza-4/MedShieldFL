import re
import csv

def extract_metrics(txt_file, csv_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    data = []
    epoch = 0
    
    for i in range(len(lines)):
        # Match the pattern for Loss, Accuracy, and AUC
        match = re.search(r'Loss: ([\d.]+), Accuracy: ([\d.]+), AUC: ([\d.]+)', lines[i])
        if match:
            loss, acc, auc = map(float, match.groups())
            epoch += 1
            data.append([epoch, auc, acc, loss])
    
    # Save to CSV
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "AUC", "Accuracy", "Loss"])
        writer.writerows(data)
    
    print(f"Extracted data saved to {csv_file}")

# Usage
extract_metrics('/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/isic/fixed_noise/0.84, e=10/client_1/log_file.txt', 'output.csv')