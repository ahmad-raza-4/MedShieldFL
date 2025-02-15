import re
import csv

def extract_metrics_from_txt(file_path, output_csv):
    # Define regex patterns
    epoch_pattern = re.compile(r'Global Epoch \(Round\): (\d+)')
    auc_pattern = re.compile(r'AUC: ([0-9\.]+)')
    acc_pattern = re.compile(r'Accuracy: ([0-9\.]+)')
    loss_pattern = re.compile(r'Loss: ([0-9\.]+)')

    # Read the file and extract data
    with open(file_path, 'r') as file:
        data = file.read()
    
    epoch = epoch_pattern.search(data)
    auc = auc_pattern.search(data)
    acc = acc_pattern.search(data)
    loss = loss_pattern.search(data)
    
    if epoch and auc and acc and loss:
        extracted_data = [
            int(epoch.group(1)), 
            float(auc.group(1)), 
            float(acc.group(1)), 
            float(loss.group(1))
        ]
    else:
        print("Error: Some values could not be extracted.")
        return
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "AUC", "Accuracy", "Loss"])
        writer.writerow(extracted_data)
    
    print(f"Data successfully written to {output_csv}")

# Example usage
extract_metrics_from_txt('log_file.txt', 'output.csv')
