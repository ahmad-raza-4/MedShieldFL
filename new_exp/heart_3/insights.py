import pandas as pd
import numpy as np

# Step 1: Read the CSV data
try:
    results = pd.read_csv("results_fed_heart_disease.csv")
except FileNotFoundError:
    print("Error: The file 'results_fed_heart_disease.csv' was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: The file 'results_fed_heart_disease.csv' is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: The file 'results_fed_heart_disease.csv' is malformed.")
    exit(1)

# Step 2: Rename the 'perf' column to 'Performance' for clarity
if 'perf' in results.columns:
    results = results.rename(columns={"perf": "Performance"})
else:
    print("Error: The expected column 'perf' is not in the CSV file.")
    exit(1)

# Step 3: Verify that necessary columns exist
required_columns = {'d', 'e', 'Performance'}
if not required_columns.issubset(results.columns):
    missing = required_columns - set(results.columns)
    print(f"Error: Missing columns in the CSV file: {missing}")
    exit(1)

# Optional: Handle missing or NaN values if necessary
# For example, you might want to exclude them or fill them with a default value
# Here, we'll exclude any rows with NaN in 'd', 'e', or 'Performance'
results_clean = results.dropna(subset=['d', 'e', 'Performance'])

# Step 4: Group by 'd' and 'e' and calculate the mean Performance
avg_perf = results_clean.groupby(['d', 'e'])['Performance'].mean().reset_index()

# Optional: Sort the results for better readability
avg_perf = avg_perf.sort_values(by=['d', 'e'])

# Step 5: Print the results
print("Average Performance for each combination of Delta (d) and Epsilon (e):\n")
print(avg_perf.to_string(index=False))

# Optional: Present the results as a pivot table
pivot_table = avg_perf.pivot(index='d', columns='e', values='Performance')

print("\nPivot Table of Average Performance:")
print(pivot_table.to_string(float_format="{:.4f}".format))
