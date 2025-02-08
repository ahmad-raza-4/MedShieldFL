import re
import csv

def parse_text_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    data = []
    epoch = 0  # Epoch counter
    
    for line in lines:
        # Extract AUC, Accuracy, and Loss
        auc_match = re.search(r'AUC:\s([\d\.]+)', line)
        acc_match = re.search(r'Accuracy:\s([\d\.]+)', line)
        loss_match = re.search(r'Loss:\s([\d\.]+)', line)
        
        if loss_match:
            loss = float(loss_match.group(1))
            auc = float(auc_match.group(1)) if auc_match else None
            acc = float(acc_match.group(1)) if acc_match else None
            data.append([epoch, auc, loss, acc])
            epoch += 1  # Increment epoch count for each set of loss values
    
    # Write to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "AUC", "Loss", "Accuracy"])
        writer.writerows(data)
    
    print(f"CSV file saved as {output_file}")

# Example usage
parse_text_file("/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/alzheimer's/client_side_adaptive_clipping/3.79, e=1.0/client_1/log_file.txt", "output.csv")
